# This file tests if the machine learning model can be trained.

from data.data_loader import load_data
from models.ml_models import train_random_forest
from preprocessing.preprocessing import preprocess_data


def test_train_random_forest():
    """Check that the model is created and data is split."""
    df = load_data()
    X, y, _ = preprocess_data(df)

    model, X_train, X_test, y_train, y_test = train_random_forest(X, y)

    # Check that the model is created.
    assert model is not None

    # Check that train and test data have rows.
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

    # X and y should match in each split.
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)

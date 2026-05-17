# This file tests whether the machine learning model can be trained.

# load_data is imported so the test can load the real dataset.
from data.data_loader import load_data
# train_random_forest is imported so the test can train the model.
from models.ml_models import train_random_forest
# preprocess_data is imported so the model receives processed input.
from preprocessing.preprocessing import preprocess_data


def test_train_random_forest():
    """Check that the model is created and data is split."""
    # Load the dataset.
    df = load_data()
    # Preprocess the dataset into features and target values.
    X, y, _ = preprocess_data(df)

    # Train the model and collect the split datasets.
    model, X_train, X_test, y_train, y_test = train_random_forest(X, y)

    # Check that the model object was created.
    assert model is not None

    # Check that the training features have rows.
    assert X_train.shape[0] > 0
    # Check that the testing features have rows.
    assert X_test.shape[0] > 0
    # Check that the training target has values.
    assert len(y_train) > 0
    # Check that the testing target has values.
    assert len(y_test) > 0

    # Check that training features and target have matching row counts.
    assert X_train.shape[0] == len(y_train)
    # Check that testing features and target have matching row counts.
    assert X_test.shape[0] == len(y_test)

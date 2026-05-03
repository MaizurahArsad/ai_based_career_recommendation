# This file tests if model evaluation gives results.

from data.data_loader import load_data
from evaluation.evaluation import evaluate_model
from models.ml_models import train_random_forest
from preprocessing.preprocessing import preprocess_data


def test_evaluate_model():
    """Check that evaluation returns accuracy and confusion matrix."""
    df = load_data()
    X, y, _ = preprocess_data(df)
    model, X_train, X_test, y_train, y_test = train_random_forest(X, y)

    accuracy, report, matrix = evaluate_model(model, X_test, y_test)

    # Check that accuracy is returned.
    assert accuracy is not None

    # Check that confusion matrix exists.
    assert matrix is not None

    # Confusion matrix should have rows and columns.
    assert matrix.shape[0] > 0
    assert matrix.shape[1] > 0

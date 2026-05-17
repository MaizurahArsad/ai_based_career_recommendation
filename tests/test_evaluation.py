# This file tests whether model evaluation returns valid results.

# load_data is imported so the test can use the real dataset.
from data.data_loader import load_data
# evaluate_model is imported so the test can check the evaluation step.
from evaluation.evaluation import evaluate_model
# train_random_forest is imported so the test can create a trained model.
from models.ml_models import train_random_forest
# preprocess_data is imported so the input is ready for the model.
from preprocessing.preprocessing import preprocess_data


def test_evaluate_model():
    """Check that evaluation returns accuracy and confusion matrix."""
    # Load the original dataset.
    df = load_data()
    # Preprocess the dataset into features and target values.
    X, y, _ = preprocess_data(df)
    # Train the model so it can be evaluated.
    model, X_train, X_test, y_train, y_test = train_random_forest(X, y)

    # Run the evaluation step and collect the results.
    accuracy, report, matrix = evaluate_model(model, X_test, y_test)

    # Check that an accuracy value was returned.
    assert accuracy is not None

    # Check that the text report was returned.
    assert report is not None

    # Check that the confusion matrix was returned.
    assert matrix is not None

    # Check that the confusion matrix has at least one row.
    assert matrix.shape[0] > 0
    # Check that the confusion matrix has at least one column.
    assert matrix.shape[1] > 0

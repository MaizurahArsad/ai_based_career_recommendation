# This file checks how well a trained model performs.

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test):
    """Test the model and return accuracy, report, and confusion matrix."""
    # Use the model to predict answers for the test data.
    y_pred = model.predict(X_test)

    # Compare the predictions with the real answers.
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(report)

    return accuracy, report, matrix

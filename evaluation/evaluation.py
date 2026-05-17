# This file checks how well a trained model performs.

# accuracy_score calculates the overall accuracy.
# classification_report shows precision, recall, and F1-score.
# confusion_matrix shows correct and incorrect predictions by class.
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test):
    """Test the model and return accuracy, report, and confusion matrix."""
    # Use the trained model to predict labels for the test feature data.
    y_pred = model.predict(X_test)

    # accuracy stores the percentage of correct predictions.
    accuracy = accuracy_score(y_test, y_pred)
    # report stores the detailed text report for each class.
    report = classification_report(y_test, y_pred)
    # matrix stores the confusion matrix for the predictions.
    matrix = confusion_matrix(y_test, y_pred)

    # Print the main evaluation results in the terminal.
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(report)

    # Return the evaluation results so other files can reuse them.
    return accuracy, report, matrix

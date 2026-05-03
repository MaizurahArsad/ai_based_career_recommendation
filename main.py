# This file runs the full machine learning pipeline.
# It loads data, prepares it, trains a model, evaluates it, and shows plots.

from data.data_loader import load_data
from evaluation.evaluation import evaluate_model
from models.ml_models import train_random_forest
from preprocessing.preprocessing import preprocess_data
from utils.visualization import (
    plot_career_distribution,
    plot_confusion_matrix,
    plot_feature_importance,
)


def run_pipeline():
    # Run each main step of the machine learning process.
    """Run the complete career recommendation ML training pipeline."""
    data = load_data()
    X, y, _ = preprocess_data(data)
    model, X_train, X_test, y_train, y_test = train_random_forest(X, y)

    return data, model, X_train, X_test, y_train, y_test, X.columns


def main():
    """Run the pipeline and print the final results."""
    # Start the pipeline and get the trained model and test data.
    data, model, X_train, X_test, y_train, y_test, feature_names = run_pipeline()

    # Check how well the model performs.
    _, _, matrix = evaluate_model(model, X_test, y_test)

    # Show useful charts for the model and data.
    plot_confusion_matrix(matrix)
    plot_feature_importance(model, feature_names)
    plot_career_distribution(data)

    # Print a short summary for the user.
    print("ML pipeline completed successfully!")
    print("Evaluation completed successfully!")
    print("Visualization completed successfully!")
    print("Model:", model.__class__.__name__)
    print("Training features shape:", X_train.shape)
    print("Testing features shape:", X_test.shape)
    print("Training target size:", len(y_train))
    print("Testing target size:", len(y_test))

# Controls when program runs
if __name__ == "__main__":
    main()

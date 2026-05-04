# This file runs the full machine learning pipeline.
# It loads data, prepares it, trains a model, evaluates it, and shows plots.

from data.data_loader import load_data
from evaluation.evaluation import evaluate_model
from models.ml_models import train_random_forest
from preprocessing.preprocessing import preprocess_data
from utils.visualization import (
    plot_age_distribution,
    plot_all_graphs_summary,
    plot_career_distribution,
    plot_confusion_matrix,
    plot_education_counts,
    plot_interest_frequencies,
    plot_score_distribution,
    plot_skill_frequencies,
    plot_skill_frequencies_horizontal,
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

    # Show and save all notebook charts.
    plot_age_distribution(data)
    plot_education_counts(data)
    plot_career_distribution(data)
    plot_score_distribution(data)
    plot_skill_frequencies(data)
    plot_skill_frequencies_horizontal(data)
    plot_interest_frequencies(data)
    plot_confusion_matrix(matrix)
    plot_all_graphs_summary(data, matrix)

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

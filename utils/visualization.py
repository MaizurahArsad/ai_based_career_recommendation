# This file creates charts to help understand the model and data.

import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(matrix):
    """Show a heatmap of correct and wrong predictions."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names):
    """Show which features are most important to the model."""
    # Get the importance score for each feature.
    importances = model.feature_importances_

    # Sort features from most important to least important.
    sorted_features = sorted(
        zip(feature_names, importances),
        key=lambda item: item[1],
        reverse=True,
    )
    features, scores = zip(*sorted_features)

    plt.figure(figsize=(10, 6))
    plt.bar(features, scores)
    plt.title("Feature Importance")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_career_distribution(df):
    """Show how many times each career appears in the dataset."""
    # Count each recommended career.
    career_counts = df["Recommended_Career"].value_counts()

    plt.figure(figsize=(10, 6))
    plt.bar(career_counts.index, career_counts.values)
    plt.title("Career Distribution")
    plt.xlabel("Recommended Career")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

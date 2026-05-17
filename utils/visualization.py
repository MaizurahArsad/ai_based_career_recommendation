# This file creates charts to help understand the data and model.

# os is used to create folders and build save paths for plot images.
import os

# matplotlib.pyplot is used to create and show charts.
import matplotlib.pyplot as plt
# pandas is used here to build value count results as a Series.
import pandas as pd
# seaborn is used to create cleaner statistical visualizations.
import seaborn as sns


# PLOTS_DIR stores the folder where all plot images will be saved.
PLOTS_DIR = "outputs/ai_career_plots"
# Create the output folder if it does not already exist.
os.makedirs("outputs/ai_career_plots", exist_ok=True)
# Set a simple chart style for all seaborn plots.
sns.set(style="whitegrid")


def count_items(series):
    """Split semicolon-separated values and count each item."""
    # Remove missing values and split each text entry by semicolon.
    items = series.dropna().apply(lambda x: x.split(";"))
    # Flatten the nested lists and remove extra spaces from each item.
    flat_list = [item.strip() for sublist in items for item in sublist]
    # Return the frequency count for each cleaned item.
    return pd.Series(flat_list).value_counts()


def _save_plot(filename):
    """Save the current plot before showing it."""
    # Save the active plot into the output folder with the given filename.
    plt.savefig(os.path.join(PLOTS_DIR, filename))


def plot_age_distribution(df):
    """Show the age distribution."""
    # Create a new figure for the age chart.
    plt.figure(figsize=(8, 4))
    # Draw a histogram with a density curve for the Age column.
    sns.histplot(df["Age"], bins=15, kde=True, color="skyblue")
    # Set the chart title.
    plt.title("Age Distribution")
    # Label the x-axis.
    plt.xlabel("Age")
    # Label the y-axis.
    plt.ylabel("Frequency")
    # Save the chart image.
    _save_plot("age_distribution.png")
    # Show the chart on screen.
    plt.show()


def plot_education_counts(df):
    """Show how many candidates are in each education level."""
    # Create a new figure for the education count chart.
    plt.figure(figsize=(6, 4))
    # Draw a bar chart for education categories.
    sns.countplot(x="Education", data=df, palette="Set2")
    # Set the chart title.
    plt.title("Education Level Counts")
    # Label the x-axis.
    plt.xlabel("Education Level")
    # Label the y-axis.
    plt.ylabel("Count")
    # Save the chart image.
    _save_plot("education_counts.png")
    # Show the chart on screen.
    plt.show()


def plot_career_distribution(df):
    """Show how many times each career appears in the dataset."""
    # Create a new figure for the career distribution chart.
    plt.figure(figsize=(10, 6))
    # Draw a horizontal count plot ordered by the most frequent careers.
    sns.countplot(
        y="Recommended_Career",
        data=df,
        order=df["Recommended_Career"].value_counts().index,
        palette="Set3",
    )
    # Set the chart title.
    plt.title("Distribution of Recommended Careers")
    # Label the x-axis.
    plt.xlabel("Count")
    # Label the y-axis.
    plt.ylabel("Recommended Career")
    # Save the chart image.
    _save_plot("career_distribution.png")
    # Show the chart on screen.
    plt.show()


def plot_score_distribution(df):
    """Show the recommendation score distribution."""
    # Create a new figure for the score chart.
    plt.figure(figsize=(8, 4))
    # Draw a histogram with a density curve for recommendation scores.
    sns.histplot(df["Recommendation_Score"], bins=10, kde=True, color="olive")
    # Set the chart title.
    plt.title("Recommendation Score Distribution")
    # Label the x-axis.
    plt.xlabel("Recommendation Score")
    # Label the y-axis.
    plt.ylabel("Frequency")
    # Save the chart image.
    _save_plot("score_distribution.png")
    # Show the chart on screen.
    plt.show()


def plot_skill_frequencies(df):
    """Show skill frequencies as vertical bars."""
    # Count how many times each skill appears in the dataset.
    skills_count = count_items(df["Skills"])

    # Create a new figure for the vertical skill chart.
    plt.figure(figsize=(10, 4))
    # Draw a bar chart for skill frequencies.
    sns.barplot(x=skills_count.index, y=skills_count.values, palette="viridis")
    # Rotate x-axis labels to make long names easier to read.
    plt.xticks(rotation=45)
    # Set the chart title.
    plt.title("Skill Frequencies")
    # Label the x-axis.
    plt.xlabel("Skill")
    # Label the y-axis.
    plt.ylabel("Frequency")
    # Save the chart image.
    _save_plot("skill_frequencies.png")
    # Show the chart on screen.
    plt.show()


def plot_skill_frequencies_horizontal(df):
    """Show skill frequencies as horizontal bars."""
    # Count how many times each skill appears in the dataset.
    skills_count = count_items(df["Skills"])

    # Create a new figure for the horizontal skill chart.
    plt.figure(figsize=(18, 14))
    # Draw a horizontal bar chart for skill frequencies.
    sns.barplot(x=skills_count.values, y=skills_count.index, palette="viridis")
    # Set the chart title.
    plt.title("Skill Frequencies")
    # Label the x-axis.
    plt.xlabel("Frequency")
    # Label the y-axis.
    plt.ylabel("Skill")
    # Adjust the layout so labels fit better.
    plt.tight_layout()
    # Save the chart image.
    _save_plot("skill_frequencies_horizontal.png")
    # Show the chart on screen.
    plt.show()


def plot_interest_frequencies(df):
    """Show interest frequencies as horizontal bars."""
    # Count how many times each interest appears in the dataset.
    interests_count = count_items(df["Interests"])

    # Create a new figure for the interest chart.
    plt.figure(figsize=(12, 8))
    # Draw a horizontal bar chart for interest frequencies.
    sns.barplot(x=interests_count.values, y=interests_count.index, palette="magma")
    # Set the chart title.
    plt.title("Interest Frequencies")
    # Label the x-axis.
    plt.xlabel("Frequency")
    # Label the y-axis.
    plt.ylabel("Interest")
    # Adjust the layout so labels fit better.
    plt.tight_layout()
    # Save the chart image.
    _save_plot("interest_frequencies.png")
    # Show the chart on screen.
    plt.show()


def plot_confusion_matrix(matrix):
    """Show a heatmap of correct and wrong predictions."""
    # Create a new figure for the confusion matrix.
    plt.figure(figsize=(10, 8))
    # Draw the matrix as a heatmap with cell values shown.
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    # Label the x-axis.
    plt.xlabel("Predicted")
    # Label the y-axis.
    plt.ylabel("Actual")
    # Set the chart title.
    plt.title("Confusion Matrix")
    # Adjust the layout so labels fit well.
    plt.tight_layout()
    # Save the chart image.
    _save_plot("confusion_matrix.png")
    # Show the chart on screen.
    plt.show()


def plot_all_graphs_summary(df, matrix):
    """Show all 8 plots in one summary image."""
    # Count the skill values for the summary charts.
    skills_count = count_items(df["Skills"])
    # Count the interest values for the summary charts.
    interests_count = count_items(df["Interests"])

    # Create one large figure with 8 subplot areas.
    fig, axes = plt.subplots(4, 2, figsize=(18, 24))
    # Flatten the axes array so it is easier to use by index.
    axes = axes.flatten()

    # Draw the age distribution chart in the first subplot.
    sns.histplot(df["Age"], bins=15, kde=True, color="skyblue", ax=axes[0])
    axes[0].set_title("Age Distribution")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Frequency")

    # Draw the education count chart in the second subplot.
    sns.countplot(x="Education", data=df, palette="Set2", ax=axes[1])
    axes[1].set_title("Education Counts")
    axes[1].set_xlabel("Education Level")
    axes[1].set_ylabel("Count")

    # Draw the career distribution chart in the third subplot.
    sns.countplot(
        y="Recommended_Career",
        data=df,
        order=df["Recommended_Career"].value_counts().index,
        palette="Set3",
        ax=axes[2],
    )
    axes[2].set_title("Career Distribution")
    axes[2].set_xlabel("Count")
    axes[2].set_ylabel("Recommended Career")

    # Draw the recommendation score chart in the fourth subplot.
    sns.histplot(
        df["Recommendation_Score"],
        bins=10,
        kde=True,
        color="olive",
        ax=axes[3],
    )
    axes[3].set_title("Score Distribution")
    axes[3].set_xlabel("Recommendation Score")
    axes[3].set_ylabel("Frequency")

    # Draw the vertical skill frequency chart in the fifth subplot.
    sns.barplot(x=skills_count.index, y=skills_count.values, palette="viridis", ax=axes[4])
    axes[4].set_title("Skill Frequencies")
    axes[4].set_xlabel("Skill")
    axes[4].set_ylabel("Frequency")
    axes[4].tick_params(axis="x", rotation=45)

    # Draw the horizontal skill frequency chart in the sixth subplot.
    sns.barplot(x=skills_count.values, y=skills_count.index, palette="viridis", ax=axes[5])
    axes[5].set_title("Skill Frequencies Horizontal")
    axes[5].set_xlabel("Frequency")
    axes[5].set_ylabel("Skill")

    # Draw the interest frequency chart in the seventh subplot.
    sns.barplot(x=interests_count.values, y=interests_count.index, palette="magma", ax=axes[6])
    axes[6].set_title("Interest Frequencies")
    axes[6].set_xlabel("Frequency")
    axes[6].set_ylabel("Interest")

    # Draw the confusion matrix in the eighth subplot.
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=axes[7])
    axes[7].set_title("Confusion Matrix")
    axes[7].set_xlabel("Predicted")
    axes[7].set_ylabel("Actual")

    # Adjust the layout so the full summary figure is readable.
    plt.tight_layout()
    # Save the final summary figure.
    _save_plot("all_plots_summary.png")
    # Show the combined summary figure.
    plt.show()

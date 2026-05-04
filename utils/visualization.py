# This file creates charts to help understand the data and model.

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PLOTS_DIR = "outputs/ai_career_plots"
os.makedirs("outputs/ai_career_plots", exist_ok=True)
sns.set(style="whitegrid")


def count_items(series):
    """Split semicolon-separated values and count each item."""
    items = series.dropna().apply(lambda x: x.split(";"))
    flat_list = [item.strip() for sublist in items for item in sublist]
    return pd.Series(flat_list).value_counts()


def _save_plot(filename):
    """Save the current plot before showing it."""
    plt.savefig(os.path.join(PLOTS_DIR, filename))


def plot_age_distribution(df):
    """Show the age distribution."""
    plt.figure(figsize=(8, 4))
    sns.histplot(df["Age"], bins=15, kde=True, color="skyblue")
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    _save_plot("age_distribution.png")
    plt.show()


def plot_education_counts(df):
    """Show how many candidates are in each education level."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Education", data=df, palette="Set2")
    plt.title("Education Level Counts")
    plt.xlabel("Education Level")
    plt.ylabel("Count")
    _save_plot("education_counts.png")
    plt.show()


def plot_career_distribution(df):
    """Show how many times each career appears in the dataset."""
    plt.figure(figsize=(10, 6))
    sns.countplot(
        y="Recommended_Career",
        data=df,
        order=df["Recommended_Career"].value_counts().index,
        palette="Set3",
    )
    plt.title("Distribution of Recommended Careers")
    plt.xlabel("Count")
    plt.ylabel("Recommended Career")
    _save_plot("career_distribution.png")
    plt.show()


def plot_score_distribution(df):
    """Show the recommendation score distribution."""
    plt.figure(figsize=(8, 4))
    sns.histplot(df["Recommendation_Score"], bins=10, kde=True, color="olive")
    plt.title("Recommendation Score Distribution")
    plt.xlabel("Recommendation Score")
    plt.ylabel("Frequency")
    _save_plot("score_distribution.png")
    plt.show()


def plot_skill_frequencies(df):
    """Show skill frequencies as vertical bars."""
    skills_count = count_items(df["Skills"])

    plt.figure(figsize=(10, 4))
    sns.barplot(x=skills_count.index, y=skills_count.values, palette="viridis")
    plt.xticks(rotation=45)
    plt.title("Skill Frequencies")
    plt.xlabel("Skill")
    plt.ylabel("Frequency")
    _save_plot("skill_frequencies.png")
    plt.show()


def plot_skill_frequencies_horizontal(df):
    """Show skill frequencies as horizontal bars."""
    skills_count = count_items(df["Skills"])

    plt.figure(figsize=(18, 14))
    sns.barplot(x=skills_count.values, y=skills_count.index, palette="viridis")
    plt.title("Skill Frequencies")
    plt.xlabel("Frequency")
    plt.ylabel("Skill")
    plt.tight_layout()
    _save_plot("skill_frequencies_horizontal.png")
    plt.show()


def plot_interest_frequencies(df):
    """Show interest frequencies as horizontal bars."""
    interests_count = count_items(df["Interests"])

    plt.figure(figsize=(12, 8))
    sns.barplot(x=interests_count.values, y=interests_count.index, palette="magma")
    plt.title("Interest Frequencies")
    plt.xlabel("Frequency")
    plt.ylabel("Interest")
    plt.tight_layout()
    _save_plot("interest_frequencies.png")
    plt.show()


def plot_confusion_matrix(matrix):
    """Show a heatmap of correct and wrong predictions."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    _save_plot("confusion_matrix.png")
    plt.show()


def plot_all_graphs_summary(df, matrix):
    """Show all 8 plots in one summary image."""
    skills_count = count_items(df["Skills"])
    interests_count = count_items(df["Interests"])

    fig, axes = plt.subplots(4, 2, figsize=(18, 24))
    axes = axes.flatten()

    sns.histplot(df["Age"], bins=15, kde=True, color="skyblue", ax=axes[0])
    axes[0].set_title("Age Distribution")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Frequency")

    sns.countplot(x="Education", data=df, palette="Set2", ax=axes[1])
    axes[1].set_title("Education Counts")
    axes[1].set_xlabel("Education Level")
    axes[1].set_ylabel("Count")

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

    sns.barplot(x=skills_count.index, y=skills_count.values, palette="viridis", ax=axes[4])
    axes[4].set_title("Skill Frequencies")
    axes[4].set_xlabel("Skill")
    axes[4].set_ylabel("Frequency")
    axes[4].tick_params(axis="x", rotation=45)

    sns.barplot(x=skills_count.values, y=skills_count.index, palette="viridis", ax=axes[5])
    axes[5].set_title("Skill Frequencies Horizontal")
    axes[5].set_xlabel("Frequency")
    axes[5].set_ylabel("Skill")

    sns.barplot(x=interests_count.values, y=interests_count.index, palette="magma", ax=axes[6])
    axes[6].set_title("Interest Frequencies")
    axes[6].set_xlabel("Frequency")
    axes[6].set_ylabel("Interest")

    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=axes[7])
    axes[7].set_title("Confusion Matrix")
    axes[7].set_xlabel("Predicted")
    axes[7].set_ylabel("Actual")

    plt.tight_layout()
    _save_plot("all_plots_summary.png")
    plt.show()


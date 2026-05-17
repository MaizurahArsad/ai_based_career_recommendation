# This file runs the full machine learning pipeline.
# It loads data, prepares it, trains a model, evaluates it, and shows plots.

# pandas is used for DataFrame handling and table preparation.
import pandas as pd
# tabulate is used to print clean tables in the terminal.
from tabulate import tabulate

# load_data reads the dataset from the data folder.
from data.data_loader import load_data
# evaluate_model checks how well the trained model performs.
from evaluation.evaluation import evaluate_model
# train_random_forest trains the Random Forest classifier.
from models.ml_models import train_random_forest
# preprocess_data prepares the dataset for machine learning.
from preprocessing.preprocessing import preprocess_data
# These functions create and save the charts used in the project.
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

# TABLE_FORMAT sets the table style used by tabulate.
TABLE_FORMAT = "github"
# PREVIEW_ROWS controls how many rows are shown in preview tables.
PREVIEW_ROWS = 5
# TOP_ITEM_LIMIT controls how many top skills and interests are shown.
TOP_ITEM_LIMIT = 10


def _print_section(title: str) -> None:
    """Print a visible section title in the terminal."""
    # Print a blank line and the section title.
    print(f"\n{title}")
    # Print a line under the title for easier reading.
    print("=" * len(title))


def _print_dataframe_table(title: str, frame: pd.DataFrame) -> None:
    """Print a DataFrame in table form."""
    # Print the section header before the table.
    _print_section(title)
    # Stop early if the table has no rows.
    if frame.empty:
        # Tell the user that there is no data to show.
        print("No rows to display.")
        return

    # Print the DataFrame as a formatted terminal table.
    print(tabulate(frame, headers="keys", tablefmt=TABLE_FORMAT, showindex=True))


def print_table(title: str, frame: pd.DataFrame) -> None:
    """Print a table using the shared terminal table format."""
    # Reuse the shared DataFrame table printer.
    _print_dataframe_table(title, frame)


def _print_series_table(title: str, series: pd.Series, value_column: str) -> None:
    """Print a Series as a two-column table."""
    # Convert the Series into a regular DataFrame.
    frame = series.reset_index()
    # Rename the columns so the table labels are clearer.
    frame.columns = ["Value", value_column]
    # Print the converted table.
    _print_dataframe_table(title, frame)


def _print_dataset_info(data: pd.DataFrame) -> None:
    """Print DataFrame info details in table form."""
    # Build a table that shows each column, non-null count, and data type.
    info_frame = pd.DataFrame(
        {
            "Column": data.columns,
            "Non-Null Count": data.count().values,
            "Dtype": data.dtypes.astype(str).values,
        }
    )
    # Print the dataset info table.
    _print_dataframe_table("Dataset Info", info_frame)

    # Print the total number of rows in the dataset.
    print(f"Total entries: {len(data)}")
    # Print the memory used by the dataset.
    print(f"Memory usage: {data.memory_usage(deep=True).sum()} bytes")


def print_datatype_summary(data: pd.DataFrame) -> None:
    """Print a summary of column counts by dtype."""
    # Count how many columns belong to each data type.
    dtype_counts = data.dtypes.astype(str).value_counts()
    # Convert the data type counts into a DataFrame table.
    dtype_frame = dtype_counts.rename_axis("Dtype").reset_index(name="Column Count")
    # Print the datatype summary table.
    print_table("Datatype Summary", dtype_frame)
    # Print a short one-line summary for quick reading.
    print(", ".join(f"{dtype}({count})" for dtype, count in dtype_counts.items()))


def _build_missing_values_table(data: pd.DataFrame) -> pd.DataFrame:
    """Build a table showing missing values by column."""
    # Count missing values in each column.
    missing_counts = data.isna().sum()
    # Convert missing counts into percentages.
    missing_percentages = (missing_counts / len(data) * 100).round(2)
    # Return the missing value information as a DataFrame.
    return pd.DataFrame(
        {
            "Column": missing_counts.index,
            "Missing Values": missing_counts.values,
            "Missing %": missing_percentages.values,
        }
    )


def _build_multi_value_counts(series: pd.Series, column_name: str) -> pd.DataFrame:
    """Split semicolon-delimited values and count the most frequent items."""
    # Fill missing values, convert to text, and split each row by semicolon.
    exploded = (
        series.fillna("")
        .astype(str)
        .str.split(";")
        .explode()
        .str.strip()
    )
    # Count the most common non-empty items.
    counts = exploded[exploded.ne("")].value_counts().head(TOP_ITEM_LIMIT)
    # Return the counts as a two-column DataFrame.
    return counts.rename_axis(column_name).reset_index(name="Count")


def _print_dataset_overview(data: pd.DataFrame) -> None:
    """Print Kaggle-style dataset summaries in the terminal."""
    # Print only the raw dataset section when processed data is not available yet.
    print_notebook_outputs(data, None, None)


def _print_processed_data_overview(X: pd.DataFrame, y: pd.Series) -> None:
    """Print processed features and target output tables."""
    # Show the first few rows of the processed feature matrix.
    _print_dataframe_table("Processed Features Preview", X.head(PREVIEW_ROWS))

    # Print a section title for the feature matrix shape.
    _print_section("Feature Matrix Shape")
    # Print the number of rows in the feature matrix.
    print(f"Rows: {X.shape[0]}")
    # Print the number of columns in the feature matrix.
    print(f"Columns: {X.shape[1]}")
    # Print the full shape tuple.
    print(f"Shape: {X.shape}")

    # Print the feature column names as a table.
    _print_dataframe_table(
        "Feature Column Names",
        pd.DataFrame({"Feature Column": X.columns}),
    )
    # Print the first few encoded target values.
    _print_dataframe_table("Target Preview", y.head(PREVIEW_ROWS).to_frame())


def print_notebook_outputs(data: pd.DataFrame, X: pd.DataFrame | None, y: pd.Series | None) -> None:
    """Print notebook-style dataset and processed-data outputs."""
    # Print the first few rows of the raw dataset.
    print_table("Dataset Preview", data.head(PREVIEW_ROWS))

    # Print a section title for the dataset shape.
    _print_section("Dataset Shape")
    # Print the number of dataset rows.
    print(f"Rows: {data.shape[0]}")
    # Print the number of dataset columns.
    print(f"Columns: {data.shape[1]}")
    # Print the full shape tuple.
    print(f"Shape: {data.shape}")

    # Print detailed dataset info.
    _print_dataset_info(data)
    # Print the count of columns by data type.
    print_datatype_summary(data)
    # Print a summary description table for the dataset.
    print_table("Dataset Description", data.describe(include="all").fillna(""))
    # Print the missing value table.
    print_table("Missing Values", _build_missing_values_table(data))

    # Print counts for the Education column if it exists.
    if "Education" in data.columns:
        _print_series_table(
            "Education Counts",
            data["Education"].value_counts(dropna=False),
            "Count",
        )

    # Print counts for the Recommended_Career column if it exists.
    if "Recommended_Career" in data.columns:
        _print_series_table(
            "Recommended Career Counts",
            data["Recommended_Career"].value_counts(dropna=False),
            "Count",
        )

    # Print the most common individual skills if the Skills column exists.
    if "Skills" in data.columns:
        print_table(
            "Most Common Skills",
            _build_multi_value_counts(data["Skills"], "Skill"),
        )

    # Print the most common individual interests if the Interests column exists.
    if "Interests" in data.columns:
        print_table(
            "Most Common Interests",
            _build_multi_value_counts(data["Interests"], "Interest"),
        )

    # Print processed feature and target outputs only when X and y are available.
    if X is not None and y is not None:
        _print_processed_data_overview(X, y)


def run_pipeline():
    """Run the complete career recommendation ML training pipeline."""
    # Load the raw dataset from the CSV file.
    data = load_data()
    # Preprocess the raw dataset into model features and target labels.
    X, y, _ = preprocess_data(data)
    # Print the notebook-style terminal outputs for the dataset and processed data.
    print_notebook_outputs(data, X, y)

    # Train the Random Forest model using the processed data.
    model, X_train, X_test, y_train, y_test = train_random_forest(X, y)

    # Return all important objects needed by the main function.
    return data, model, X_train, X_test, y_train, y_test, X.columns


def main():
    """Run the pipeline and print the final results."""
    # Run the full pipeline and collect the main outputs.
    data, model, X_train, X_test, y_train, y_test, feature_names = run_pipeline()

    # Evaluate the model using the test data.
    _, _, matrix = evaluate_model(model, X_test, y_test)

    # Create and save the age distribution chart.
    plot_age_distribution(data)
    # Create and save the education count chart.
    plot_education_counts(data)
    # Create and save the career distribution chart.
    plot_career_distribution(data)
    # Create and save the score distribution chart.
    plot_score_distribution(data)
    # Create and save the vertical skill frequency chart.
    plot_skill_frequencies(data)
    # Create and save the horizontal skill frequency chart.
    plot_skill_frequencies_horizontal(data)
    # Create and save the interest frequency chart.
    plot_interest_frequencies(data)
    # Create and save the confusion matrix chart.
    plot_confusion_matrix(matrix)
    # Create and save the combined summary chart.
    plot_all_graphs_summary(data, matrix)

    # Print the final pipeline summary section.
    _print_section("Pipeline Summary")
    # Print the success message for the machine learning pipeline.
    print("ML pipeline completed successfully!")
    # Print the success message for evaluation.
    print("Evaluation completed successfully!")
    # Print the success message for visualization.
    print("Visualization completed successfully!")
    # Print the model class name.
    print("Model:", model.__class__.__name__)
    # Print the training feature shape.
    print("Training features shape:", X_train.shape)
    # Print the testing feature shape.
    print("Testing features shape:", X_test.shape)
    # Print the training target size.
    print("Training target size:", len(y_train))
    # Print the testing target size.
    print("Testing target size:", len(y_test))
    # Print the number of feature columns used by the model.
    print("Feature column count:", len(feature_names))


# Run the main function only when this file is executed directly.
if __name__ == "__main__":
    main()

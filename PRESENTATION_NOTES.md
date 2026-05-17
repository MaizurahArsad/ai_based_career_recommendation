# Presentation Notes: AI-Based Career Recommendation System

## 1. Repository Overview

### What this project does

This project is a simple AI-based career recommendation system.
It reads a dataset of candidate profiles and tries to predict a suitable career using machine learning.

In simple words:

- the system reads candidate data
- it prepares the data for a machine learning model
- it trains a model to learn patterns
- it tests the model
- it shows the results in terminal tables and charts

### Main objective of the project

The main objective is to build a small end-to-end machine learning pipeline that can recommend a career based on:

- age
- education
- skills
- interests
- recommendation score

This project is also meant for academic presentation, so it focuses on showing the complete workflow clearly, not only the final accuracy.

### Full workflow from dataset loading until final output

The workflow is:

1. `main.py` starts the project.
2. `data/data_loader.py` loads the CSV dataset from the `data/` folder.
3. `preprocessing/preprocessing.py` selects useful columns and converts text categories into numbers.
4. `main.py` prints notebook-style terminal tables such as dataset preview, datatype summary, missing values, skill counts, and feature matrix shape.
5. `models/ml_models.py` splits the data into training and testing sets.
6. `models/ml_models.py` trains a `RandomForestClassifier`.
7. `evaluation/evaluation.py` predicts test results and calculates accuracy, classification report, and confusion matrix.
8. `utils/visualization.py` creates charts and saves them into `outputs/ai_career_plots/`.
9. `main.py` prints the final pipeline summary.

### How the files and folders are connected

#### Core runtime flow

- `main.py`
  - imports the data loader
  - imports preprocessing
  - imports model training
  - imports evaluation
  - imports visualization

- `data/data_loader.py`
  - reads the dataset file from `data/`
  - returns a pandas DataFrame to `main.py`

- `preprocessing/preprocessing.py`
  - receives the DataFrame from `main.py`
  - validates required columns
  - creates `X` and `y`
  - returns processed features and labels to `main.py`

- `models/ml_models.py`
  - receives `X` and `y`
  - splits them into train and test sets
  - trains the Random Forest model
  - returns the model and split data to `main.py`

- `evaluation/evaluation.py`
  - receives the trained model and testing data
  - computes accuracy, report, and confusion matrix
  - returns results to `main.py`

- `utils/visualization.py`
  - receives the raw dataset and confusion matrix
  - generates plots
  - saves images into `outputs/ai_career_plots/`

#### Supporting files

- `README.md`
  - explains the project to a reader

- `requirements.txt`
  - lists Python packages needed to run the project

- `tests/`
  - contains simple test files for data loading, preprocessing, training, and evaluation

- `ai-career-recommendation-eda-ml (2).py`
  - converted notebook-style script
  - useful for showing the project origin and the earlier Kaggle-style workflow
  - not the main script used for final execution

- `notebook/input-output-print-analysis.md`
  - notes about console print behavior in the converted notebook script

- `models/deep_learning.py`
  - placeholder file
  - not used in the current pipeline

## 2. File-by-File Explanation

### `main.py`

#### What this file is for

This is the main entry point of the whole project.
When we run `python main.py`, this file controls the full pipeline from start to finish.

#### Libraries imported and why

- `pandas`
  - used to work with DataFrames when printing tables
- `tabulate`
  - used to display clean tables in the terminal
- `load_data`
  - loads the dataset
- `evaluate_model`
  - evaluates the trained model
- `train_random_forest`
  - trains the machine learning model
- `preprocess_data`
  - prepares raw data for model training
- visualization functions
  - create and save charts

#### Important variables

- `TABLE_FORMAT = "github"`
  - makes the terminal tables look structured
- `PREVIEW_ROWS = 5`
  - shows the first 5 rows in previews
- `TOP_ITEM_LIMIT = 10`
  - limits skill and interest frequency tables to top 10 items

#### Functions and what they do

- `_print_section(title)`
  - prints a clear section title in the terminal

- `_print_dataframe_table(title, frame)`
  - prints a pandas DataFrame as a terminal table

- `print_table(title, frame)`
  - wrapper function that reuses the same table printing style

- `_print_series_table(title, series, value_column)`
  - converts a pandas Series into a table and prints it

- `_print_dataset_info(data)`
  - shows column name, non-null count, data type, total entries, and memory usage

- `print_datatype_summary(data)`
  - counts how many columns belong to each data type
  - prints a datatype table
  - prints a one-line summary like `str(5), int64(2), float64(1)`

- `_build_missing_values_table(data)`
  - builds a table of missing values by column

- `_build_multi_value_counts(series, column_name)`
  - splits semicolon-separated text such as `Skills` or `Interests`
  - counts the most common items

- `_print_dataset_overview(data)`
  - helper that prints notebook-style dataset output
  - this function exists, but in the current flow it is not actually used by `run_pipeline()`

- `_print_processed_data_overview(X, y)`
  - prints processed features, feature matrix shape, column names, and target preview

- `print_notebook_outputs(data, X, y)`
  - prints the main terminal outputs that replace notebook display

- `run_pipeline()`
  - loads data
  - preprocesses data
  - prints terminal output
  - trains the model
  - returns everything needed for evaluation and summary

- `main()`
  - calls `run_pipeline()`
  - evaluates the model
  - generates all visualizations
  - prints final summary

#### How this file connects to other files

This file is the coordinator.
It does not contain the actual loading logic, preprocessing logic, training logic, or evaluation logic by itself.
Instead, it imports those steps from other modules and runs them in the correct order.

---

### `data/data_loader.py`

#### What this file is for

This file is responsible for loading the dataset from the CSV file in the `data/` folder.

#### Libraries imported and why

- `Path` from `pathlib`
  - used to build a reliable file path to the dataset
- `pandas`
  - used to read the CSV file into a DataFrame

#### Important variables

- `DATA_PATH`
  - stores the full file path of the dataset

#### Functions and what they do

- `load_data()`
  - checks if the dataset exists
  - raises an error if the file is missing
  - reads the CSV file
  - prints a small success message and shape
  - returns the DataFrame

#### How this file connects to other files

`main.py` imports `load_data()` from this file.
Without this file, the whole pipeline cannot start because there is no dataset to process.

---

### `preprocessing/preprocessing.py`

#### What this file is for

This file prepares the raw dataset for machine learning.
It selects input columns, extracts the target column, checks required fields, and converts text categories into numbers.

#### Libraries imported and why

- `pandas`
  - used for DataFrame and Series handling
- `LabelEncoder`
  - used to convert text values into numeric labels

#### Important variables

- `FEATURE_COLUMNS`
  - list of input columns used for model training
  - `Age`, `Education`, `Skills`, `Interests`, `Recommendation_Score`

- `TARGET_COLUMN`
  - output column to predict
  - `Recommended_Career`

- `CATEGORICAL_COLUMNS`
  - columns that need encoding because they are text
  - `Education`, `Skills`, `Interests`

#### Functions and what they do

- `preprocess_data(data)`
  - validates the dataset columns
  - creates `X` for features
  - creates `y` for labels
  - label-encodes categorical columns in `X`
  - label-encodes the target column in `y`
  - returns `X`, `y`, and encoders

- `_validate_columns(data)`
  - checks that all required columns exist
  - raises a clear error if any are missing

#### How this file connects to other files

`main.py` sends the loaded DataFrame to this file.
`models/ml_models.py` later receives the processed `X` and `y`.

---

### `models/ml_models.py`

#### What this file is for

This file handles machine learning model training.

#### Libraries imported and why

- `RandomForestClassifier`
  - the main model used in the project
- `train_test_split`
  - splits data into training and testing parts

#### Functions and what they do

- `train_random_forest(X, y)`
  - splits the data
  - creates a Random Forest model
  - trains it on training data
  - returns the model and split datasets

#### Important settings

- `test_size=0.2`
  - 20% of the data is used for testing

- `random_state=42`
  - makes the split and model more reproducible

- `n_estimators=100`
  - uses 100 trees in the Random Forest

#### How this file connects to other files

It receives processed data from `main.py` and returns a trained model plus train/test data.
Then `evaluation/evaluation.py` uses that model.

---

### `evaluation/evaluation.py`

#### What this file is for

This file evaluates the trained model.

#### Libraries imported and why

- `accuracy_score`
  - calculates overall accuracy
- `classification_report`
  - shows precision, recall, and F1-score
- `confusion_matrix`
  - compares predicted classes with true classes

#### Functions and what they do

- `evaluate_model(model, X_test, y_test)`
  - predicts labels for test data
  - computes accuracy
  - computes classification report
  - computes confusion matrix
  - prints results
  - returns results

#### How this file connects to other files

`main.py` sends the trained model and test data here.
`utils/visualization.py` later uses the confusion matrix to create a heatmap.

---

### `utils/visualization.py`

#### What this file is for

This file creates all charts used in the project.

#### Libraries imported and why

- `os`
  - creates the output folder and save paths
- `matplotlib.pyplot`
  - basic plotting control
- `pandas`
  - supports value counting output
- `seaborn`
  - cleaner and easier statistical plots

#### Important variables

- `PLOTS_DIR`
  - folder where plot images are saved

#### Functions and what they do

- `count_items(series)`
  - splits semicolon-separated text into individual items
  - counts frequencies

- `_save_plot(filename)`
  - saves the current plot to the output folder

- `plot_age_distribution(df)`
  - histogram of ages

- `plot_education_counts(df)`
  - count plot of education levels

- `plot_career_distribution(df)`
  - count plot of recommended careers

- `plot_score_distribution(df)`
  - histogram of recommendation scores

- `plot_skill_frequencies(df)`
  - vertical bar chart of most common skills

- `plot_skill_frequencies_horizontal(df)`
  - horizontal bar chart of skills

- `plot_interest_frequencies(df)`
  - horizontal bar chart of interests

- `plot_confusion_matrix(matrix)`
  - heatmap of model prediction results

- `plot_all_graphs_summary(df, matrix)`
  - combines all major plots into one summary image

#### How this file connects to other files

`main.py` calls these functions after evaluation.
This file depends on the raw dataset and the confusion matrix.

---

### `requirements.txt`

#### What this file is for

This file lists the Python packages needed to run the project.

#### Libraries included

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scipy`
- `tabulate`

#### Why it matters

It helps another user install the exact tools needed using:

```bash
pip install -r requirements.txt
```

#### Notes

- `scipy` is not used in the final `main.py` pipeline, but it appears in the converted notebook script.
- This is acceptable, but it also shows the project contains some legacy notebook-related dependencies.

---

### `README.md`

#### What this file is for

This file explains the project to someone opening the repository.

#### What it contains

- project title
- project purpose
- dataset column explanation
- ML pipeline explanation
- terminal output explanation
- visualization explanation
- project folder structure
- setup and run instructions

#### Why it matters

For submission, the README acts like the project manual.
It is useful for the lecturer, classmates, or anyone who wants to run the code.

---

### Dataset file inside `data/`

#### File

- `data/AI-based Career Recommendation System (2).csv`

#### What it is

This is the dataset used for the entire project.

#### What it contains

The dataset has:

- 200 rows
- 8 columns

Columns:

- `CandidateID`
- `Name`
- `Age`
- `Education`
- `Skills`
- `Interests`
- `Recommended_Career`
- `Recommendation_Score`

#### Important characteristics

- `Skills` and `Interests` are semicolon-separated text fields
- `Recommended_Career` is the target column
- based on the current terminal output, there are no missing values
- the data is synthetic, not real-world collected data

#### How it connects to the project

This file is loaded by `data_loader.py`.
All preprocessing, model training, evaluation, and visualization depend on this dataset.

---

### Other important repository files

#### `ai-career-recommendation-eda-ml (2).py`

- This is a converted notebook-style Python script.
- It shows the earlier Kaggle-style workflow.
- It uses a richer preprocessing idea than the final main pipeline:
  - `CountVectorizer`
  - sparse matrix combination
  - combined text features
- It is useful for presentation context, but it is not the main runnable submission script.

#### `models/deep_learning.py`

- Placeholder only
- no model logic inside
- safe to mention as future expansion, not current functionality

#### `tests/`

- simple checks for:
  - data loading
  - preprocessing
  - model training
  - evaluation

These tests are useful to show that the basic pipeline parts work.

## 3. Line-by-Line Code Explanation

This section explains the important logic section by section in simple words.

### `main.py` line-by-line explanation by function

#### Imports and constants

- `import pandas as pd`
  - needed because tables are created using pandas DataFrames
  - if removed, table preparation code would fail

- `from tabulate import tabulate`
  - used to print nice terminal tables
  - if removed, the output would no longer show the clean table format

- imports from `data`, `preprocessing`, `models`, `evaluation`, `utils`
  - connect all project parts into one workflow
  - if removed, `main.py` would not know how to load data, preprocess, train, evaluate, or plot

- constants:
  - `TABLE_FORMAT`
  - `PREVIEW_ROWS`
  - `TOP_ITEM_LIMIT`
  - these control the display style
  - if removed, the code would either fail or become harder to manage

#### `_print_section(title)`

Logic:

1. print a blank line and the title
2. print a line of `=` under the title

Why needed:

- makes the terminal output easy to read

If removed:

- the output still works, but it becomes messy and harder to present

#### `_print_dataframe_table(title, frame)`

Logic:

1. print the section header
2. check whether the DataFrame is empty
3. if empty, print a message
4. otherwise, print the DataFrame as a `tabulate` table

Why needed:

- central place for printing tables consistently

If removed:

- many other output functions would break because they depend on this helper

#### `print_table(title, frame)`

Logic:

1. calls `_print_dataframe_table()`

Why needed:

- simple wrapper for consistent naming
- easier to reuse in notebook-style output code

If removed:

- only small refactoring would be needed, but current code uses it directly

#### `_print_series_table(title, series, value_column)`

Logic:

1. convert a Series into a DataFrame
2. rename columns
3. print it as a table

Why needed:

- `value_counts()` returns a Series, not a normal table
- this makes it presentable

If removed:

- education and career count output would be harder to print cleanly

#### `_print_dataset_info(data)`

Logic:

1. build a table with column names
2. count non-null values
3. show data types
4. print total entries
5. print memory usage

Why needed:

- gives a script-friendly replacement for `df.info()`

If removed:

- the lecturer cannot see the dataset structure as clearly in terminal output

#### `print_datatype_summary(data)`

Logic:

1. count how many columns belong to each data type
2. build a table
3. print the table
4. print a quick one-line summary

Why needed:

- gives a fast summary of dataset column types

If removed:

- datatype explanation becomes less clear during presentation

#### `_build_missing_values_table(data)`

Logic:

1. count missing values in each column
2. calculate missing percentages
3. return them as a DataFrame

Why needed:

- missing values affect preprocessing and model quality

If removed:

- we lose transparency about data quality

#### `_build_multi_value_counts(series, column_name)`

Logic:

1. replace missing values with empty string
2. convert values to string
3. split each row by semicolon
4. explode the list into separate rows
5. remove extra spaces
6. remove empty values
7. count the most common items
8. return as a table

Why needed:

- `Skills` and `Interests` contain multiple values in one cell
- this function turns them into meaningful frequency counts

If removed:

- skill and interest analysis would be much weaker

#### `_print_processed_data_overview(X, y)`

Logic:

1. print first rows of processed features
2. print feature matrix shape
3. print feature column names
4. print target preview

Why needed:

- shows what the model actually sees after preprocessing

If removed:

- the lecturer might ask how raw data became model input and you would have less evidence on screen

#### `print_notebook_outputs(data, X, y)`

Logic:

1. print raw dataset preview
2. print dataset shape
3. print dataset info
4. print datatype summary
5. print description table
6. print missing values
7. print education counts if column exists
8. print recommended career counts if column exists
9. print most common skills if column exists
10. print most common interests if column exists
11. if `X` and `y` exist, print processed data overview

Why needed:

- this is the terminal replacement for notebook display cells

If removed:

- the project still trains and evaluates, but loses its presentation-friendly output

#### `run_pipeline()`

Logic:

1. call `load_data()`
2. call `preprocess_data(data)`
3. print notebook-style outputs
4. call `train_random_forest(X, y)`
5. return results

Why needed:

- groups the core pipeline steps into one reusable function

If removed:

- `main()` would have to do everything directly and become less organized

#### `main()`

Logic:

1. call `run_pipeline()`
2. evaluate the model
3. create all plots
4. print final summary

Why needed:

- main execution controller

If removed:

- there is no single entry point for `python main.py`

---

### `data/data_loader.py` line-by-line explanation

#### Imports

- `Path`
  - used to build the dataset path safely

- `pandas`
  - used to read the CSV file

#### `DATA_PATH`

- creates the dataset file path relative to the current file
- this is better than hardcoding an absolute path

If removed:

- the loader would not know where to find the dataset

#### `load_data()`

1. check if the dataset file exists
2. if not, raise `FileNotFoundError`
3. read the CSV file with `pd.read_csv()`
4. print success message
5. print dataset shape
6. return the DataFrame

Why needed:

- this is the only source of dataset loading in the main runtime

If the existence check is removed:

- the program would fail later with a less clear error

If `read_csv()` is removed:

- no data would be loaded at all

---

### `preprocessing/preprocessing.py` line-by-line explanation

#### `FEATURE_COLUMNS`

- tells the model which inputs to use

If removed:

- feature selection becomes unclear and code breaks

#### `TARGET_COLUMN`

- tells the model what to predict

If removed:

- the code cannot build labels

#### `CATEGORICAL_COLUMNS`

- lists which input columns are text-based and need encoding

If removed:

- the loop that encodes text features would fail or become unclear

#### `preprocess_data(data)`

1. call `_validate_columns(data)`
   - makes sure required columns exist
2. copy feature columns into `X`
3. copy target column into `y`
4. create `encoders = {}`
5. loop through categorical columns
6. create a `LabelEncoder` for each one
7. replace missing values with `"Unknown"`
8. convert column to string
9. transform text into numbers
10. create another `LabelEncoder` for the target
11. encode the target values
12. store target encoder in the dictionary
13. return `X`, `y`, and encoders

Why needed:

- machine learning models cannot use raw text labels directly

If encoding is removed:

- Random Forest training would fail because strings would remain in the input

#### `_validate_columns(data)`

1. combine all required input and target columns
2. compare them with actual dataset columns
3. collect missing ones
4. raise an error if any are missing

Why needed:

- prevents silent failure and confusing bugs

If removed:

- code may break later in a less clear way

---

### `models/ml_models.py` line-by-line explanation

#### `train_random_forest(X, y)`

1. call `train_test_split(X, y, test_size=0.2, random_state=42)`
   - creates training and testing data
2. create `RandomForestClassifier(n_estimators=100, random_state=42)`
3. call `model.fit(X_train, y_train)`
4. return model and split datasets

Why needed:

- training and testing must be separated to evaluate performance honestly

If train/test split is removed:

- the model may be evaluated on the same data it learned from
- that would give misleading results

If `fit()` is removed:

- the model would never learn and evaluation would fail

---

### `evaluation/evaluation.py` line-by-line explanation

#### `evaluate_model(model, X_test, y_test)`

1. call `model.predict(X_test)`
2. store predictions in `y_pred`
3. calculate accuracy
4. calculate classification report
5. calculate confusion matrix
6. print accuracy
7. print report
8. return all evaluation results

Why needed:

- training alone is not enough
- we must check whether the model performs well on unseen data

If prediction is removed:

- there is nothing to evaluate

If confusion matrix is removed:

- visual analysis of classification performance becomes weaker

---

### `utils/visualization.py` line-by-line explanation

#### Global setup

- `PLOTS_DIR`
  - stores output folder name

- `os.makedirs(..., exist_ok=True)`
  - ensures the output folder exists before saving plots

- `sns.set(style="whitegrid")`
  - gives all charts a cleaner style

If folder creation is removed:

- plot saving may fail if the folder does not exist

#### `count_items(series)`

1. remove missing values
2. split each cell by semicolon
3. flatten all lists into one list
4. trim spaces
5. return value counts

Why needed:

- turns text lists into countable items

#### `_save_plot(filename)`

1. save the current plot using `plt.savefig()`

Why needed:

- preserves plot images in the output folder

#### Plot functions

Each plot function follows the same pattern:

1. create a figure
2. draw a seaborn plot
3. set title and labels
4. save the plot
5. show the plot

Why needed:

- each graph explains one part of the data or model

If the plot is not saved:

- the result cannot be reviewed later

If `plt.show()` is removed:

- in some environments the chart may not appear interactively

## 4. Presentation Script

You can say this during your presentation.

### Greeting and project introduction

Good morning, sir.
Today I would like to present my project, which is an AI-based career recommendation system using Python and machine learning.

### Problem statement

The problem I focus on is that career recommendation can be difficult when we have many candidate attributes such as education, skills, and interests.
So I built a simple system that learns from a dataset and predicts a recommended career.

### Project objective

The main objective of this project is to build a complete machine learning pipeline.
The system loads candidate data, preprocesses it, trains a model, evaluates the model, and shows the results in terminal tables and graphs.

### Dataset explanation

The dataset is stored in CSV format inside the `data` folder.
It contains candidate information such as Candidate ID, Name, Age, Education, Skills, Interests, Recommended Career, and Recommendation Score.
In this project, the target column is `Recommended_Career`, because that is the value the model tries to predict.

### System workflow

The workflow starts in `main.py`.
First, the system loads the dataset using `data_loader.py`.
Next, it preprocesses the data using `preprocessing.py`.
After that, it prints notebook-style tables in the terminal so we can inspect the data.
Then it splits the data into training and testing sets, trains a Random Forest model, evaluates the results, and finally generates visualizations.

### Code structure explanation

The project is divided into folders based on function.
The `data` folder handles dataset loading.
The `preprocessing` folder prepares the dataset.
The `models` folder trains the machine learning model.
The `evaluation` folder checks model performance.
The `utils` folder contains visualization functions.
Then `main.py` connects all these parts together.

### Preprocessing explanation

In preprocessing, I first validate that all required columns exist.
Then I select the feature columns and the target column.
The text columns, especially Education, Skills, and Interests, are encoded into numeric values using `LabelEncoder` because machine learning models cannot work directly with raw text.

### Model training explanation

For model training, I use `RandomForestClassifier`.
Before training, the data is split into training and testing sets using `train_test_split`.
The model learns from the training set and later is tested on the testing set.

### Evaluation explanation

After training, the system predicts the testing data.
Then it calculates accuracy, classification report, and confusion matrix.
Accuracy gives a simple overall score, while the classification report and confusion matrix show more detailed performance.

### Visualization explanation

The project also generates graphs such as age distribution, education counts, career distribution, score distribution, skill frequency, interest frequency, and confusion matrix.
These charts help us understand both the dataset and the model results more clearly.

### Terminal output explanation

Because this project runs as a normal Python script, not inside Kaggle Notebook, I use `tabulate` to show dataframe-style tables in the terminal.
The output includes dataset preview, shape, info, datatype summary, missing values, common skills, common interests, processed features, and evaluation results.

### Final conclusion

In conclusion, this project demonstrates a complete and simple AI-based career recommendation workflow.
It may not be a perfect recommendation system, but it shows how machine learning can be used from raw data until final prediction, evaluation, and visualization.
Thank you.

## 5. Lecturer Q&A Preparation

### Why did you use Random Forest?

Because Random Forest is simple, powerful, and works well for classification problems.
It can also handle mixed feature patterns better than a very basic single model.

### Why did you split the dataset into training and testing?

To evaluate the model fairly.
If I test on the same data used for training, the result may look better than the true performance.

### Why is preprocessing needed?

Raw data is not ready for machine learning.
The model needs the correct columns, clean structure, and numeric values instead of raw text.

### Why did you encode the education column?

Because `Education` is text such as Bachelor's or Master's.
Machine learning models need numbers, so I converted those categories into numeric labels.

### Why did you split skills and interests?

Because one row may contain multiple skills or multiple interests separated by semicolons.
Splitting them makes it possible to count and visualize individual items.

### Why does the terminal output look different from Kaggle?

Because Kaggle and Jupyter can display DataFrames in notebook cells automatically.
This project runs as a Python script in IntelliJ terminal, so I use `tabulate` to create readable tables in the terminal.

### Why are there warning messages from sklearn?

The current model does not predict every class equally well.
Some classes have very few test examples or no predicted samples, so sklearn warns that some precision or recall values are undefined.

### Why is the accuracy around the current value?

The dataset has many career classes and only 200 rows.
Also, the final main pipeline uses simple label encoding for complex text columns like Skills and Interests.
That reduces how much information the model can learn.

### What is feature matrix shape?

It means the size of the input data used by the model.
For example, `(200, 5)` means 200 rows and 5 feature columns.

### What is target variable?

The target variable is the output the model tries to predict.
In this project, it is `Recommended_Career`.

### What is the difference between features and labels?

Features are the input values used to make predictions.
Labels are the correct answers the model tries to learn.

In this project:

- features = Age, Education, Skills, Interests, Recommendation_Score
- label = Recommended_Career

### What happens inside `main.py`?

It controls the full workflow:

1. load the data
2. preprocess the data
3. print terminal outputs
4. train the model
5. evaluate the model
6. generate visualizations
7. print final summary

### What happens inside each folder?

- `data`
  - dataset loading
- `preprocessing`
  - data preparation
- `models`
  - model training
- `evaluation`
  - model evaluation
- `utils`
  - helper functions such as visualization
- `tests`
  - basic checking files

### Why use `requirements.txt`?

It makes setup easier.
Anyone can install the needed libraries with one command.

### Why use `tabulate`?

To print clean tables in the terminal.
Without it, the output would be much harder to read during presentation.

### What is the purpose of each graph?

- Age distribution
  - shows the age spread of candidates
- Education counts
  - shows how many candidates belong to each education level
- Career distribution
  - shows which careers appear most often
- Score distribution
  - shows how recommendation scores are distributed
- Skill frequency
  - shows which skills appear most often
- Interest frequency
  - shows which interests appear most often
- Confusion matrix
  - shows where the model predicts correctly and incorrectly
- Summary chart
  - combines multiple graphs into one figure

### What are the limitations of the system?

- the dataset is synthetic
- the dataset is small
- there are many classes
- the final main pipeline simplifies `Skills` and `Interests` using label encoding
- there is no hyperparameter tuning
- there is no user interface

### How can this project be improved in the future?

- use better text processing for `Skills` and `Interests`
- try TF-IDF or multi-label feature engineering
- compare multiple models
- tune model parameters
- use larger and more realistic data
- add a user interface
- save the trained model for reuse

## 6. Quick Revision Notes

### Important terms and meanings

- Dataset
  - collection of data used by the project
- Feature
  - input variable used for prediction
- Label / Target
  - output variable to predict
- Preprocessing
  - preparing raw data for machine learning
- Train/test split
  - dividing data into learning and evaluation parts
- Accuracy
  - percentage of correct predictions
- Confusion matrix
  - table showing correct and wrong predictions by class
- Label encoding
  - converting text categories into numbers
- DataFrame
  - table structure used by pandas
- Terminal table
  - script-based display of table data using `tabulate`

### Important functions and their purpose

- `load_data()`
  - loads CSV dataset
- `preprocess_data(data)`
  - prepares features and labels
- `_validate_columns(data)`
  - checks required columns
- `train_random_forest(X, y)`
  - splits data and trains model
- `evaluate_model(model, X_test, y_test)`
  - computes accuracy, report, confusion matrix
- `print_notebook_outputs(data, X, y)`
  - prints presentation-style terminal tables
- `plot_age_distribution(df)`
  - age histogram
- `plot_confusion_matrix(matrix)`
  - confusion matrix heatmap
- `plot_all_graphs_summary(df, matrix)`
  - combined plot summary

### Important outputs and what they mean

- Dataset preview
  - first rows of raw data
- Dataset shape
  - number of rows and columns
- Dataset info
  - column names, non-null counts, data types
- Datatype summary
  - how many columns are text, integers, floats
- Missing values
  - checks data quality
- Education counts
  - class distribution for education
- Recommended career counts
  - class distribution for target values
- Most common skills/interests
  - frequency analysis of multi-value text fields
- Processed features preview
  - what the model sees after preprocessing
- Feature matrix shape
  - size of model input
- Target preview
  - encoded output values
- Accuracy
  - overall prediction score
- Classification report
  - detailed performance per class
- Confusion matrix
  - correct vs wrong predictions

### Common errors and warnings and how to explain them

- `FileNotFoundError`
  - happens if dataset path is wrong or file is missing
  - explanation: the loader cannot find the CSV file

- sklearn `UndefinedMetricWarning`
  - happens when some classes have no predicted samples or no true samples in a split
  - explanation: class distribution is uneven and the model is not strong enough on every class

- seaborn `FutureWarning`
  - related to plotting API usage, not a model logic failure
  - explanation: current plotting still works, but future seaborn versions may require small syntax updates

- matplotlib `FigureCanvasAgg is non-interactive`
  - appears when using a non-interactive backend during scripted runs
  - explanation: plots are still saved correctly, but they may not open interactively in that mode

## 7. Honest Weak Points and Improvement Notes

This section is important because the lecturer may ask what is weak in the project.

### Weak point 1: final pipeline is simpler than the notebook version

The converted notebook script uses better text handling:

- `CountVectorizer`
- combined text features
- sparse matrix stacking

But the final `main.py` pipeline uses simple label encoding for `Skills` and `Interests`.

Why this matters:

- label encoding treats full text combinations like category labels
- it does not capture individual skills or interests as rich features
- this likely reduces model quality

### Weak point 2: dataset is synthetic

This means:

- it is useful for demonstration
- but it may not represent real career recommendation complexity

### Weak point 3: dataset is small compared to number of classes

There are many target classes but only 200 rows.
This makes prediction harder and can cause unstable class performance.

### Weak point 4: many classes are imbalanced

Some careers appear more often than others.
That can cause the model to predict common classes more easily than rare classes.

### Weak point 5: warnings show the model is weak on some classes

The sklearn warnings are not just technical noise.
They show that some classes are not predicted well enough.

### Weak point 6: `_print_dataset_overview()` is currently not used in the real flow

The function exists in `main.py`, but `run_pipeline()` directly calls `print_notebook_outputs()` instead.

This is not a breaking issue, but it is slightly redundant code.

### Weak point 7: `models/deep_learning.py` is only a placeholder

This is acceptable, but if asked, you should say:

> The file is reserved for future expansion and is not part of the current working pipeline.

### Weak point 8: tests are basic

The tests only check that:

- data loads
- preprocessing returns output
- model trains
- evaluation returns results

They do not deeply verify:

- output correctness
- model quality
- plot file creation
- error edge cases

### Weak point 9: no model saving

The trained model is not saved to disk.
Every run trains the model again.

### Weak point 10: no end-user interface

The project works well as an academic Python script, but it is not yet a real application for public users.

## 8. Strong Points You Can Emphasize

If you want to balance the weak points, you can mention these strengths:

- clear modular structure
- full ML workflow from loading to visualization
- readable terminal outputs for presentation
- plots are automatically saved
- beginner-friendly code organization
- reproducible training using fixed `random_state`
- simple tests are included

## 9. Short Final Summary for Yourself

If you need a very short answer during presentation, use this:

> This project is a modular Python machine learning system that predicts `Recommended_Career` from candidate information.
> The workflow starts from CSV loading, then preprocessing, then Random Forest training, then evaluation, and finally visualization.
> The main script is `main.py`, while other folders separate loading, preprocessing, training, evaluation, and plotting.
> The current system works correctly, but its main limitations are the small synthetic dataset and the simple encoding of text-based columns.

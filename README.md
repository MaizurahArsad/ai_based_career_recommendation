# AI-Based Career Recommendation System Using Random Forest

## Student Information

- Student Name: Maizurah Binti Arsad
- Student ID: BS23110083
- Course Code: SF35803 Computer Programming 2
- Session: 2025/2026
- Semester: 2

## Project Overview

This project is an AI-based career recommendation system.
It uses candidate information such as age, education, skills, interests, and recommendation score to predict a suitable career path.

The main purpose of this project is to show a simple machine learning workflow in Python.
It demonstrates how data can be loaded, prepared, used to train a model, evaluated, and visualized in a clear academic project format.

## Project Purpose

This project was built to:

- explore a career recommendation dataset
- prepare the data for machine learning
- train a classification model
- predict recommended careers
- show results in both terminal tables and charts

## Dataset Columns

The dataset used in this project contains these main columns:

- `CandidateID`: unique ID for each candidate
- `Name`: candidate name
- `Age`: candidate age
- `Education`: education level of the candidate
- `Skills`: skills listed for the candidate
- `Interests`: interests listed for the candidate
- `Recommended_Career`: target career label to predict
- `Recommendation_Score`: score related to the recommendation

## Machine Learning Pipeline

The project follows this simple pipeline:

1. Load the dataset from the `data/` folder.
2. Preprocess the data by selecting important columns and encoding text values.
3. Train a `RandomForestClassifier` model.
4. Evaluate the model using accuracy, classification report, and confusion matrix.
5. Generate visualizations for better understanding of the dataset and model results.

## Terminal Outputs

When you run `python main.py`, the script prints notebook-style terminal tables so the project is easy to present without Jupyter Notebook.

The terminal output includes:

- dataset preview
- dataset shape
- dataset info
- datatype summary
- dataset description
- missing values table
- education counts
- recommended career counts
- most common skills
- most common interests
- processed feature preview
- feature matrix shape
- feature column names
- target preview
- model evaluation results

## Visual Output Charts

The project also generates and saves charts in `outputs/ai_career_plots/`.

These charts include:

- age distribution
- education level counts
- recommended career distribution
- recommendation score distribution
- skill frequency chart
- horizontal skill frequency chart
- interest frequency chart
- confusion matrix heatmap
- combined summary chart

## Why Terminal Tables Are Used

This version of the project runs as a normal Python script, not as a Kaggle Notebook or Jupyter Notebook.
Because of that, table-style terminal output is used to show dataframe summaries in a clear and readable way inside the IntelliJ terminal.

## Project Folder Structure

```text
ai_based_career_recommendation/
|-- data/
|   |-- AI-based Career Recommendation System (2).csv
|   `-- data_loader.py
|-- evaluation/
|   `-- evaluation.py
|-- models/
|   |-- deep_learning.py
|   `-- ml_models.py
|-- notebook/
|   `-- input-output-print-analysis.md
|-- outputs/
|   `-- ai_career_plots/
|-- preprocessing/
|   `-- preprocessing.py
|-- tests/
|   |-- test_data_loader.py
|   |-- test_evaluation.py
|   |-- test_ml_models.py
|   `-- test_preprocessing.py
|-- utils/
|   `-- visualization.py
|-- ai-career-recommendation-eda-ml (2).py
|-- main.py
|-- README.md
`-- requirements.txt
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd ai_based_career_recommendation
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
.venv\Scripts\activate
```

### 4. Install the required libraries

```bash
pip install -r requirements.txt
```

### 5. Run the project

```bash
python main.py
```

## Main Technologies Used

- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- tabulate

## Submission Note

This repository is prepared as a script-based academic submission.
The machine learning flow, terminal outputs, and saved visualizations are all designed to run directly from `main.py`.

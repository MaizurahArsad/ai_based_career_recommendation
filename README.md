# AI-Based Career Recommendation System

## Student Information

- Student Name: Maizurah Binti Arsad
- Student ID: BS23110083
- Course Code: SF35803 Computer Programming 2
- Session: 2025/2026
- Semester: 2

## Project Description

This project is a simple machine learning project.
It recommends careers based on candidate information.

The project uses:

- Age
- Education
- Skills
- Interests
- Recommendation Score

The model used is `RandomForestClassifier`.

## What This Project Does

- Load data
- Preprocess data
- Train model
- Evaluate model
- Visualize results

## Workflow Steps

### Data Loading

The project loads the dataset from the `data/` folder.
The data is read using pandas.

### Preprocessing

The project prepares the data before training.
Text values are changed into numbers so the model can use them.

### Training

The project trains a machine learning model.
It uses `RandomForestClassifier` to learn from the data.

### Evaluation

The project checks how well the model works.
It shows accuracy, classification report, and confusion matrix.

### Visualization

The project shows charts for better understanding.
It shows the confusion matrix, feature importance, and career distribution.

## Technologies Used

- Python
- pandas
- scikit-learn
- matplotlib
- seaborn

## How To Run

Run this command in the project folder:

```bash
python main.py
```

## Project Structure

- `data/` -> dataset and data loader
- `preprocessing/` -> prepare data
- `models/` -> train model
- `evaluation/` -> check performance
- `utils/` -> visualization
- `notebooks/` -> original notebook
- `tests/` -> test files
- `main.py` -> run everything

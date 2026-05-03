# AI-Based Career Recommendation System

## Project Description

This project is a simple career recommendation system.
It uses candidate data to predict a suitable career.
The project is built for learning machine learning in Python.

## What This Project Does

- Loads the career dataset
- Prepares the data for machine learning
- Trains a Random Forest model
- Checks how well the model works
- Shows charts for results and data

## Workflow Steps

1. Data loading  
   The project reads the dataset from the `data` folder.

2. Preprocessing  
   The data is cleaned and text values are changed into numbers.

3. Training  
   A Random Forest model is trained using the prepared data.

4. Evaluation  
   The model is tested using accuracy, classification report, and confusion matrix.

5. Visualization  
   Charts are shown for the confusion matrix, feature importance, and career distribution.

## Technologies Used

- Python
- pandas
- scikit-learn
- matplotlib
- seaborn

## How To Run

Run this command:

```bash
python main.py
```

## Project Structure

- `data/`  
  Contains the dataset and data loading code.

- `preprocessing/`  
  Contains code to prepare the data.

- `models/`  
  Contains code to train the machine learning model.

- `evaluation/`  
  Contains code to check model performance.

- `utils/`  
  Contains chart and visualization code.

- `notebook/`  
  Contains the original notebook work.

- `tests/`  
  Contains test files for the project.

- `main.py`  
  Runs the full project pipeline.

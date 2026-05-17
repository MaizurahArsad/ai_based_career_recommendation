# Quick Presentation Script

## 1. 3-Minute Presentation Script

Good morning, sir.
Today I would like to present my project, which is an AI-based career recommendation system using Python and machine learning.

The main idea of this project is to predict a suitable career based on candidate information.
The input data includes age, education, skills, interests, and recommendation score.
The output is the predicted recommended career.

This project starts with a CSV dataset stored in the `data` folder.
First, the system loads the dataset using `data_loader.py`.
After that, the preprocessing step selects the important columns and converts text-based values into numeric form so the machine learning model can use them.

Next, the processed data is split into training data and testing data.
I use a Random Forest Classifier as the main model.
The model learns patterns from the training set and then predicts the careers in the testing set.

After training, the system evaluates the model using accuracy, classification report, and confusion matrix.
This helps me see how well the model performs.

Besides the model result, the project also prints clear terminal tables such as dataset preview, dataset shape, datatype summary, missing values, most common skills, and most common interests.
This is useful because this project runs as a Python script, not as a Kaggle or Jupyter notebook.

The project also creates visualizations such as age distribution, education counts, career distribution, skill frequencies, interest frequencies, score distribution, and confusion matrix.
These graphs are saved in the `outputs` folder.

In conclusion, this project shows a complete machine learning workflow, starting from loading data until evaluation and visualization.
It is simple, modular, and suitable for academic presentation.
Thank you.

## 2. 5-Minute Presentation Script

Good morning, sir.
Today I would like to present my project, which is called an AI-based career recommendation system using Random Forest.

The problem I focus on is career recommendation.
Sometimes it is difficult to match a person with a suitable career when many factors are involved, such as education, skills, and interests.
So this project is designed to show how machine learning can learn from a dataset and predict a recommended career.

The dataset is stored as a CSV file in the `data` folder.
It contains columns such as Candidate ID, Name, Age, Education, Skills, Interests, Recommended Career, and Recommendation Score.
In this project, the target variable is `Recommended_Career`, because that is the value the model tries to predict.

The full workflow starts in `main.py`.
This file is the main controller of the whole project.
It calls the data loading, preprocessing, model training, evaluation, and visualization functions.

The first step is data loading.
The file `data/data_loader.py` reads the dataset using pandas and returns it as a DataFrame.

The second step is preprocessing.
This happens in `preprocessing/preprocessing.py`.
In this step, the code validates the required columns, selects the feature columns, and separates the target column.
Then text-based columns such as Education, Skills, and Interests are converted into numeric values using label encoding.
This is necessary because machine learning models cannot directly process raw text values.

After preprocessing, the script prints notebook-style outputs in the terminal.
These include dataset preview, dataset shape, dataset info, datatype summary, dataset description, missing values, education counts, recommended career counts, most common skills, most common interests, processed feature preview, feature matrix shape, and target preview.
The tables are printed using the `tabulate` library so they look clean in the IntelliJ terminal.

The third step is model training.
This happens in `models/ml_models.py`.
The code splits the processed data into training and testing sets using `train_test_split`.
I use 80 percent for training and 20 percent for testing.
Then the model used is `RandomForestClassifier`.
I chose Random Forest because it is simple, stable, and suitable for classification tasks.

The fourth step is evaluation.
This happens in `evaluation/evaluation.py`.
The model predicts the testing data, then the code calculates accuracy, classification report, and confusion matrix.
Accuracy gives the overall performance, while the classification report and confusion matrix show more detail for each class.

The fifth step is visualization.
This is handled by `utils/visualization.py`.
It generates charts such as age distribution, education counts, career distribution, score distribution, skill frequencies, interest frequencies, and confusion matrix.
These charts help explain both the dataset and the model result.

The project is modular.
Each folder has one main role.
This makes the code easier to understand and easier to present.

One important point is that this project is script-based.
That is why the output looks different from Kaggle Notebook.
In Kaggle, tables are displayed automatically in notebook cells.
Here, I use terminal tables instead.

There are also some limitations.
The dataset is synthetic, the number of rows is small, and there are many target classes.
Also, the final main pipeline uses simple label encoding for text-based columns, so the model is correct as a demo system but still has room for improvement.

In conclusion, this project demonstrates a complete machine learning pipeline from raw data until final evaluation and visualization.
It may be simple, but it shows the full process clearly and is suitable for academic submission.
Thank you.

## 3. Short Explanation for Each Folder

### `data/`

This folder stores the dataset and the file that loads the dataset.

### `preprocessing/`

This folder prepares the raw dataset before model training.
It selects columns, checks required columns, and encodes text values into numbers.

### `models/`

This folder contains machine learning model code.
Right now, the main used file is `ml_models.py`, which trains the Random Forest model.

### `evaluation/`

This folder checks how well the trained model performs.
It calculates accuracy, classification report, and confusion matrix.

### `utils/`

This folder contains helper functions.
In this project, it mainly contains visualization functions for charts.

### `tests/`

This folder contains simple test files to check whether data loading, preprocessing, training, and evaluation work correctly.

### `outputs/`

This folder stores the generated plot images after the script runs.

### `notebook/`

This folder contains notebook-related notes from the earlier project version.

## 4. Top 20 Possible Lecturer Questions with Simple Answers

### 1. What is the main purpose of this project?

The main purpose is to build a simple machine learning system that predicts a recommended career from candidate information.

### 2. Why did you choose Random Forest?

Because Random Forest is simple, reliable, and good for classification tasks.
It is also easier to explain in an academic presentation.

### 3. Why do you need preprocessing?

Because raw data is not ready for machine learning.
The model needs selected columns and numeric input instead of text.

### 4. Why did you encode the Education column?

Because Education is text, and the model needs numbers to learn from it.

### 5. Why did you also encode Skills and Interests?

Because in the final script they are still text fields, so they also must be converted into numbers before model training.

### 6. Why did you split the dataset into training and testing?

To test the model fairly on unseen data.
If I test on training data, the result may be misleading.

### 7. What is the target variable?

The target variable is `Recommended_Career`.
It is the output the model tries to predict.

### 8. What are features?

Features are the input columns used by the model, such as Age, Education, Skills, Interests, and Recommendation Score.

### 9. What is the difference between features and labels?

Features are the inputs.
Labels are the correct answers the model should learn to predict.

### 10. What happens inside `main.py`?

It runs the full workflow: load data, preprocess data, print tables, train model, evaluate model, generate charts, and print final summary.

### 11. Why is the terminal output in table format?

Because this project runs as a Python script in IntelliJ terminal, not inside a notebook.
So I use `tabulate` to make the output easy to read.

### 12. Why does the output look different from Kaggle?

Kaggle notebooks display DataFrames automatically in cells.
A normal Python script does not do that, so I print the tables manually.

### 13. Why are there warning messages from sklearn?

Because some classes are not predicted well, and some classes may have very few samples in the test set.
So sklearn warns that some metrics are undefined.

### 14. Why is the accuracy around the current value?

Because the dataset is small, has many classes, and the final preprocessing for text columns is still simple.

### 15. What is feature matrix shape?

It means the size of the processed input data used by the model.
For example, `(200, 5)` means 200 rows and 5 feature columns.

### 16. Why did you count skills and interests separately?

Because each row may contain multiple skills and interests in one text field.
Splitting them helps show which items are most common.

### 17. What is the purpose of the confusion matrix?

It shows where the model predicts correctly and where it makes mistakes.

### 18. Why use `requirements.txt`?

It lists the libraries needed to run the project.
This makes installation easier and more consistent.

### 19. What are the limitations of the project?

The data is synthetic, the dataset is small, there are many target classes, and the final text preprocessing is still basic.

### 20. How can this project be improved?

It can be improved by using better text processing, more data, model comparison, hyperparameter tuning, and maybe a user interface.

## 5. Final Backup Explanation for Unexpected Questions

If your lecturer asks an unexpected question, you can use these safe explanation patterns.

### Backup explanation 1: About project design

I separated the project into folders so each part has one clear role.
This makes the code easier to understand, test, and explain.

### Backup explanation 2: About machine learning choice

I chose a simple and stable workflow because the goal of this project is to clearly demonstrate the full machine learning pipeline, not only to chase the highest accuracy.

### Backup explanation 3: About output style

Because this is a Python script version, I adapted the output for terminal presentation by using tables and saved charts.

### Backup explanation 4: About limitations

This project works correctly as a demonstration system, but it still has room for improvement, especially in text feature handling and dataset size.

### Backup explanation 5: If you do not know the exact answer

You can say:

> Based on my current implementation, the main reason is that the project is designed as a simple and clear academic pipeline.  
> If I continue this project, that is one area I would improve next.

### Backup explanation 6: If asked why your code is modular

You can say:

> I separated loading, preprocessing, training, evaluation, and visualization into different files so the structure is cleaner and easier to maintain.

### Backup explanation 7: If asked why your model is not perfect

You can say:

> The main reason is that the dataset is small and has many career classes, so the model does not have enough examples for every class.

### Backup explanation 8: If asked what is the strongest part of your project

You can say:

> The strongest part is that it shows the complete workflow clearly, from raw dataset until final evaluation and visual output.

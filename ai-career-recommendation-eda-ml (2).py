#!/usr/bin/env python
# coding: utf-8

# Converted from ai-career-recommendation-eda-ml (2).ipynb

# %% [markdown] Cell 1
# # AI-Based Career Recommendation System – EDA and Machine Learning Notebook
#
# #### This notebook uses the synthetic dataset [AI-based Career Recommendation System](https://www.kaggle.com/datasets/adilshamim8/ai-based-career-recommendation-system/data) to:
# - 1. Load and explore the dataset.
# - 2. Perform extensive exploratory data analysis (EDA) to understand the data distribution.
# - 3. Preprocess the data to prepare it for machine learning.
# - 4. Build and evaluate a simple classification model to predict the recommended career.
# - **Note:** The dataset is synthetic. The model and EDA steps serve as an example pipeline.
#
# ## Step 1: Import Libraries and Load Data

# %% Cell 2
# pandas is used to load and inspect tabular data.
import pandas as pd
# numpy is used for general numerical work.
import numpy as np
# train_test_split is used to divide the data into training and testing sets.
from sklearn.model_selection import train_test_split
# LabelEncoder converts text labels into numbers.
# MultiLabelBinarizer can convert multiple labels into binary columns.
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
# CountVectorizer can convert text into numeric token counts.
from sklearn.feature_extraction.text import CountVectorizer
# RandomForestClassifier is the machine learning model used in this notebook.
from sklearn.ensemble import RandomForestClassifier
# classification_report and confusion_matrix are used to evaluate the model.
from sklearn.metrics import classification_report, confusion_matrix
# These plotting helpers create the visual charts used in the notebook.
from utils.visualization import (
    count_items,
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

# %% Cell 3
# Load the dataset into a pandas DataFrame.
data = pd.read_csv("/kaggle/input/ai-based-career-recommendation-system/AI-based Career Recommendation System.csv")
# Show the first few rows to preview the dataset.
data.head()

# %% [markdown] Cell 4
# # Step 2: Data Overview
# > Let's take a look at the basic structure, columns, and summary statistics of the dataset.

# %% Cell 5
print("Dataset Shape:", data.shape)
print("\nDataset Info:")
data.info()

# %% Cell 6
# Display summary statistics for numeric columns
data.describe()

# %% [markdown] Cell 7
# # Step 3: Exploratory Data Analysis (EDA)
# ## 3.1 Distribution of Age
# > We'll look at the distribution of the `Age` column.

# %% Cell 8
plot_age_distribution(data)

# %% [markdown] Cell 9
# ## 3.2 Count of Education Levels
# > Let's examine how many candidates fall into each education category.

# %% Cell 10
plot_education_counts(data)

# %% [markdown] Cell 11
# ## 3.3 Recommended Career Distribution
# > A count plot for the `Recommended_Career` column to see the frequency of each career recommendation.

# %% Cell 12
plot_career_distribution(data)

# %% [markdown] Cell 13
# ## 3.4 Recommendation Score Distribution
# > Visualize the distribution of the `Recommendation_Score`.

# %% Cell 14
plot_score_distribution(data)

# %% [markdown] Cell 15
# ## 3.5 Skills and Interests Exploration 
# > Since `Skills` and `Interests` are stored as semicolon-separated strings, we can explore the most common items. <br>
# > We'll split the strings and count the frequency of each skill and interest.

# %% Cell 16
# Function moved to utils/visualization.py

# %% Cell 17
# Count Skills
skills_count = count_items(data["Skills"])
print("Most common skills:\n", skills_count)

# %% Cell 18
# Plot skills count
plot_skill_frequencies(data)

# %% Cell 19
# Plot as horizontal bars with more vertical space
plot_skill_frequencies_horizontal(data)

# %% Cell 20
# Count Interests
interests_count = count_items(data["Interests"])
print("Most common interests:\n", interests_count)

# %% Cell 21
# Plot interests count as horizontal bars
plot_interest_frequencies(data)

# %% [markdown] Cell 22
# # Step 4: Data Preprocessing for Machine Learning
# > Our goal is to build a model to predict `Recommended_Career` based on candidate features.
# ## 4.1 Feature Selection
# > We will use:
#  - **Age** (numeric)
#  - **Education** (categorical, will be label-encoded)
#  - **Skills** and **Interests** (text fields, which we will vectorize) <br>
# We will drop columns like `CandidateID` and `Name` as they are identifiers and not useful for prediction.

# %% Cell 23
# Select features and target variable
features = data[['Age', 'Education', 'Skills', 'Interests']]
target = data['Recommended_Career']

# %% [markdown] Cell 24
# ## 4.2 Encode the Education Column
# > Convert the categorical `Education` column into numerical labels.

# %% Cell 25
edu_encoder = LabelEncoder()
features['Education_enc'] = edu_encoder.fit_transform(features['Education'])
features.head()

# %% [markdown] Cell 26
# ## 4.3 Process Text Fields: Skills and Interests
# > We will use `CountVectorizer` to convert the semicolon-separated skills and interests into numerical features. <br>
# > First, we join the two text fields into one combined text feature.

# %% Cell 27
# Create a combined text column
features['Text_Features'] = features['Skills'].fillna('') + " " + features['Interests'].fillna('')

# Use CountVectorizer for text feature extraction
vectorizer = CountVectorizer(tokenizer=lambda x: [item.strip() for item in x.split(";") if item.strip()], 
                             lowercase=True)
text_features = vectorizer.fit_transform(features['Text_Features'])

print("Shape of text features:", text_features.shape)

# %% [markdown] Cell 28
# ## 4.4 Construct the Final Feature Matrix
# > We will combine the numeric feature `Age` and the encoded `Education_enc` with the vectorized text features. <br>
# > We use scipy's sparse matrix concatenation.

# %% Cell 29
from scipy.sparse import hstack

# Create a numeric matrix from Age and Education_enc
numeric_features = features[['Age', 'Education_enc']].values

# %% Cell 30
# hstack to combine the numeric features with the text features
from scipy import sparse
X = hstack([sparse.csr_matrix(numeric_features), text_features])

# %% Cell 31
# Encode target variable
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(target)

print("Feature matrix shape:", X.shape)
print("Target classes:", target_encoder.classes_)

# %% [markdown] Cell 32
# # Step 5: Build and Evaluate a Machine Learning Model
# >  We'll use a Random Forest Classifier for this example.
# ## 5.1 Split the Data into Training and Testing Sets

# %% Cell 33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# %% [markdown] Cell 34
# ## 5.2 Train the Random Forest Classifier

# %% Cell 35
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# %% [markdown] Cell 36
# ## 5.3 Model Evaluation
# > We'll evaluate the model using a confusion matrix and classification report.

# %% Cell 37
# Predict on the test set
y_pred = clf.predict(X_test)

# %% Cell 38
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm)
plot_all_graphs_summary(data, cm)

# %% [markdown] Cell 39
# # Step 6: Summary and Next Steps
# ### In this notebook, we:
#  - Loaded the synthetic career dataset.
#  - Explored the data through various EDA techniques.
#  - Preprocessed numeric and text features.
#  - Built a simple Random Forest model to predict the recommended career.
# ### **Next Steps:**
#  - Experiment with other feature engineering techniques (e.g., TF-IDF for text).
#  - Try different machine learning algorithms.
#  - Tune hyperparameters to improve model performance.
#  - Explore the relationship between candidate attributes and career recommendations further. <br>
# This pipeline can serve as a foundation for developing a more robust career recommendation system.

# %% [markdown] Cell 40
# ### Connect with Me  
#
# Feel free to follow me on these platforms:  
#
# [![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AdilShamim8)  
# [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/adilshamim8)  
# [![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/adil_shamim8)  

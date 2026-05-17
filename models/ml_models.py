# This file trains machine learning models.

# RandomForestClassifier is the machine learning model used in this project.
from sklearn.ensemble import RandomForestClassifier
# train_test_split divides the dataset into training and testing parts.
from sklearn.model_selection import train_test_split


def train_random_forest(X, y):
    """Split the data, train a Random Forest model, and return the results."""
    # Split the feature data and target data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Create the Random Forest model with fixed settings for repeatable results.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Train the model using the training data.
    model.fit(X_train, y_train)

    # Return the model and all split datasets for later evaluation.
    return model, X_train, X_test, y_train, y_test

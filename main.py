# misc.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- Data Loading ---
def load_boston_data():
    """
    Loads the Boston Housing dataset as a pandas DataFrame.
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

    # Split into data and target as per instructions
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]

    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    return df

# --- Data Preprocessing ---
def preprocess_data(df, target_col='MEDV', test_size=0.2, random_state=42):
    """
    Splits the data into features (X) and target (y), then performs a train-test split.
    Returns: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

# --- Model Training/Evaluation ---
def train_and_evaluate_model(model_class, model_params, X_train, X_test, y_train, y_test):
    """
    Generic function to train a scikit-learn model and evaluate its performance.
    
    Args:
        model_class: The scikit-learn model class (e.g., DecisionTreeRegressor).
        model_params: A dictionary of parameters for the model.
        X_train, X_test, y_train, y_test: Split datasets.
        
    Returns: The trained pipeline and the Mean Squared Error (MSE) score.
    """
    
    # A simple pipeline for robust use: Scaling -> Model
    # Note: For DecisionTree, scaling isn't strictly necessary, but it makes the pipeline generic
    # and ready for models like KernelRidge.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model_class(**model_params))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return pipeline, mse

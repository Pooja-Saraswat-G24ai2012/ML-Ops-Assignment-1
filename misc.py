# misc.py

import logging
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Configure module-level logger
logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    # Basic configuration if no handlers are configured by the application
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

# --- Data Loading ---
def load_boston_data():
    """
    Loads the Boston Housing dataset as a pandas DataFrame.
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    logger.info("Starting to load Boston dataset from %s", data_url)
    try:
        raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
        logger.debug("Raw dataframe loaded with shape: %s", raw_df.shape)

        # Split into data and target as per instructions
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]
        logger.debug("Data matrix shape: %s, target vector shape: %s", data.shape, target.shape)

        feature_names = [
            'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
        ]

        df = pd.DataFrame(data, columns=feature_names)
        df['MEDV'] = target
        logger.info("Finished loading dataset. Final dataframe shape: %s", df.shape)
        return df

    except Exception as e:
        logger.exception("Failed to load Boston dataset: %s", e)
        raise

# --- Data Preprocessing ---
def preprocess_data(df, target_col='MEDV', test_size=0.2, random_state=42):
    """
    Splits the data into features (X) and target (y), then performs a train-test split.
    Returns: X_train, X_test, y_train, y_test
    """
    logger.info(
        "Starting preprocessing: target_col=%s, test_size=%s, random_state=%s",
        target_col, test_size, random_state
    )

    if target_col not in df.columns:
        logger.error("Target column '%s' not found in dataframe columns: %s", target_col, df.columns.tolist())
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    logger.debug("Features shape: %s, target shape: %s", X.shape, y.shape)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(
        "Completed train-test split: X_train=%s, X_test=%s, y_train=%s, y_test=%s",
        X_train.shape, X_test.shape, y_train.shape, y_test.shape
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
    model_name = getattr(model_class, "__name__", str(model_class))
    logger.info("Preparing to train model '%s' with params: %s", model_name, model_params)
    logger.debug("X_train shape: %s, y_train shape: %s", getattr(X_train, "shape", None), getattr(y_train, "shape", None))

    try:
        # A simple pipeline for robust use: Scaling -> Model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model_class(**(model_params or {})))
        ])

        logger.info("Starting training pipeline for model '%s'.", model_name)
        pipeline.fit(X_train, y_train)
        logger.info("Model '%s' training completed.", model_name)

        # Predict and evaluate
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logger.info("Model '%s' evaluation completed.", model_name)
        logger.debug("Sample predictions: %s", y_pred[:5])

        return pipeline, mse

    except Exception as e:
        logger.exception("Training or evaluation failed for model '%s': %s", model_name, e)
        raise

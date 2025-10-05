# train.py (for DecisionTreeRegressor) with logging

import logging
import sys
from sklearn.tree import DecisionTreeRegressor
from misc import load_boston_data, preprocess_data, train_and_evaluate_model

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def main():
    try:
        logger.info("Starting training script for DecisionTreeRegressor")

        # 1. Data Loading
        logger.info("Loading Boston dataset")
        df = load_boston_data()
        logger.info("Loaded dataset with shape: %s", getattr(df, "shape", "unknown"))
        logger.debug("Dataset columns: %s", getattr(df, "columns", "unknown"))

        # 2. Data Preprocessing (Split)
        logger.info("Preprocessing data and performing train/test split")
        X_train, X_test, y_train, y_test = preprocess_data(df)
        logger.info(
            "Split sizes - X_train: %s, X_test: %s, y_train: %s, y_test: %s",
            getattr(X_train, "shape", "unknown"),
            getattr(X_test, "shape", "unknown"),
            getattr(y_train, "shape", "unknown"),
            getattr(y_test, "shape", "unknown"),
        )

        # 3. Model Training and Evaluation
        model_class = DecisionTreeRegressor
        model_params = {"random_state": 42}
        logger.info("Model: %s", model_class.__name__)
        logger.info("Model parameters: %s", model_params)

        logger.info("Starting training and evaluation")
        pipeline, mse = train_and_evaluate_model(
            model_class, model_params, X_train, X_test, y_train, y_test
        )
        logger.info("Training and evaluation completed")

        # If pipeline has a representation, log it at debug level
        try:
            logger.debug("Trained pipeline/model: %s", pipeline)
        except Exception:
            logger.debug("Trained pipeline/model representation not available")

        # 4. Display Results
        logger.info("--- Decision Tree Regressor Performance ---")
        logger.info("Average Mean Squared Error (MSE) on Test Set: %.4f", mse)

    except Exception:
        logger.exception("An error occurred during training")
        sys.exit(1)


if __name__ == "__main__":
    main()

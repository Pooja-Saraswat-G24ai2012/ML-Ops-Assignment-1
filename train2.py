# train2.py (for KernelRidge) with logging

import time
import sys
import logging

from sklearn.kernel_ridge import KernelRidge
from misc import load_boston_data, preprocess_data, train_and_evaluate_model

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting KernelRidge training script")
    start_time = time.time()

    try:
        # 1. Data Loading
        logger.info("Loading Boston dataset")
        df = load_boston_data()
        shape = getattr(df, "shape", None)
        logger.info(f"Loaded data; df.shape = {shape}")

        # 2. Data Preprocessing (Split)
        logger.info("Preprocessing and splitting data")
        X_train, X_test, y_train, y_test = preprocess_data(df)
        logger.info(
            "Data split complete: "
            f"X_train.shape={getattr(X_train, 'shape', None)}, "
            f"X_test.shape={getattr(X_test, 'shape', None)}, "
            f"y_train.shape={getattr(y_train, 'shape', None)}, "
            f"y_test.shape={getattr(y_test, 'shape', None)}"
        )

        # 3. Model Training and Evaluation
        model_class = KernelRidge
        model_params = {'alpha': 1.0, 'kernel': 'rbf', 'gamma': 0.1}
        logger.info(f"Model selected: {model_class.__name__}")
        logger.info(f"Model parameters: {model_params}")

        logger.info("Beginning training and evaluation")
        t0 = time.time()
        pipeline, mse = train_and_evaluate_model(
            model_class, model_params, X_train, X_test, y_train, y_test
        )
        t1 = time.time()
        logger.info(f"Training and evaluation finished in {t1 - t0:.3f}s")

        # 4. Display Results
        logger.info("--- Kernel Ridge Regressor Performance ---")
        logger.info(f"Average Mean Squared Error (MSE) on Test Set: {mse:.4f}")

        total_time = time.time() - start_time
        logger.info(f"Script completed in {total_time:.3f}s")

    except Exception as e:
        logger.exception("An error occurred during execution")
        raise


if __name__ == "__main__":
    main()

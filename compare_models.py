# compare.py

import subprocess
import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def run_training_script(script_name):
    """
    Executes a training script and returns the model name and its MSE score.
    """
    logger.info("--- Running %s ---", script_name)
    
    try:
        # Execute the script
        result = subprocess.run(
            ['python', script_name],
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        logger.error("Script %s failed with return code %s", script_name, e.returncode)
        logger.error("stderr: %s", e.stderr)
        return script_name, None

    output = result.stdout
    logger.info(output)
    
    # Use regular expression to find the model name and MSE score
    model_match = re.search(r'---\s*(.*?)\s*Performance\s*---', output, re.IGNORECASE)
    model_name = model_match.group(1).strip() if model_match else script_name
    
    # Look for the MSE score
    mse_match = re.search(r'Mean Squared Error \(MSE\) on Test Set:\s*(\d+\.\d+)', output)
    mse_score = float(mse_match.group(1)) if mse_match else None
    
    return model_name, mse_score

def generate_comparison_report():
    """
    Runs both training scripts and prints a comparison table.
    """
    results = []
    
    # Run DecisionTreeRegressor (train.py)
    name1, mse1 = run_training_script('train.py')
    if mse1 is not None:
        results.append((name1, mse1))
    
    # Run KernelRidge (train2.py)
    name2, mse2 = run_training_script('train2.py')
    if mse2 is not None:
        results.append((name2, mse2))
    
    # --- Print Performance Comparison Table ---
    logger.info("=" * 50)
    logger.info("      ML Model Performance Comparison Report")
    logger.info("=" * 50)
    
    if not results:
        logger.warning("Could not retrieve MSE scores from training scripts.")
        return

    # Print Header
    header = f"{'Model':<30} | {'Average MSE on Test Set':<20}"
    logger.info(header)
    logger.info("-" * 50)
    
    # Print Results
    for name, mse in results:
        logger.info(f"{name:<30} | {mse:<20.4f}")
    
    logger.info("=" * 50 + "\n")

if __name__ == "__main__":
    generate_comparison_report()

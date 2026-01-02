import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from utils import load_model, evaluate_model, get_prepared_data, get_logger, setup_logging
from config import MODEL_NAMES

# Setup logging
setup_logging(log_to_file=False)
logger = get_logger(__name__)

def main():
    logger.info("Regenerating plots for trained models...")
    
    # Load test data
    logger.info("Loading test data...")
    _, X_test, _, y_test, label_encoder, _ = get_prepared_data(use_cache=True)
    
    # List of models to process
    # MODEL_NAMES keys: knn, svm (deprecated), randomforest, logistic, xgboost, lightgbm, naive_bayes
    # Need to match keys in utils.MODEL_NAMES which usually maps 'knn' -> 'knn_model.pkl'
    # But load_model takes the specific key name.
    
    # We'll iterate through known keys.
    models_to_check = ['knn', 'naivebayes', 'randomforest', 'logistic', 'xgboost', 'lightgbm']
    
    for model_name in models_to_check:
        try:
            logger.info(f"Processing {model_name}...")
            
            # Load model
            model, _, _ = load_model(model_name)
            
            # Evaluate (this triggers save_plots=True by default inside evaluate_model)
            evaluate_model(
                model, 
                X_test, 
                y_test, 
                model_name, 
                save_plots=True, 
                label_encoder=label_encoder
            )
            logger.info(f"✓ Re-generated plots for {model_name}")
            
        except FileNotFoundError:
            logger.warning(f"Model {model_name} not found. Skipping...")
        except Exception as e:
            logger.error(f"Failed to process {model_name}: {e}")

if __name__ == "__main__":
    main()

"""
Configuration file for Diabetes Prediction Project
Contains all constants and configuration parameters
"""
from pathlib import Path

# Project root directory (2 levels up from src/)
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Data files
RAW_DATA_FILE = RAW_DATA_DIR / "diabetes_dataset.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "diabetes_dataset_processed.csv"

# Model paths - organized structure
MODELS_DIR = PROJECT_ROOT / "models"  # Root directory for all models
RESULTS_DIR = PROJECT_ROOT / "results"  # Evaluation results

# Model subdirectories - each model type has its own folder
MODEL_SUBDIRS = {
    'knn': MODELS_DIR / "knn",
    'logistic': MODELS_DIR / "logistic_regression",
    'randomforest': MODELS_DIR / "random_forest",
    'svm': MODELS_DIR / "svm",
    'xgboost': MODELS_DIR / "xgboost"
}

# Model training parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
SCORING_METRIC = 'precision_macro'  # Scoring metric for cross-validation: 'precision_macro', 'recall_macro', or 'f1_macro'

# Optuna hyperparameter optimization parameters
N_TRIALS = 50  # Default number of optimization trials
OPTUNA_TIMEOUT = None  # Timeout in seconds (None = no timeout)
OPTUNA_N_JOBS = -1  # Number of parallel jobs (-1 = all cores, 1 = sequential)
# Parallel Optuna trials require independent trials for optimal performance
# Models with early stopping or complex dependencies require sequential execution (OPTUNA_N_JOBS=1)

# Model-specific trial counts (optimized for ~10 minutes per model)
MODEL_TRIALS = {
    'knn': 25,  # KNN is fast, can do more trials
    'svm': 20,  # SVM is slower, fewer trials
    'randomforest': 20,  # RF is moderate speed
    'logistic': 25,  # Logistic Regression is fast
    'xgboost': 15  # XGBoost is slower, fewer trials needed (early stopping helps)
}

# Model names
MODEL_NAMES = {
    'knn': 'knn_model.pkl',
    'logistic': 'logisticregression_model.pkl',
    'randomforest': 'randomforest_model.pkl',
    'svm': 'svm_model.pkl',
    'xgboost': 'xgboost_model.pkl'
}

# Model weights for ensemble prediction
MODEL_WEIGHTS = {
    'knn': 50,
    'logistic': 75,
    'randomforest': 90,
    'svm': 75,
    'xgboost': 90
}

# Advanced preprocessing options (optional - for maximum performance)
USE_ADVANCED_PREPROCESSING = True  # Set to True to enable advanced preprocessing
ADVANCED_PREPROCESSING_OPTIONS = {
    'handle_outliers': True,  # Detect and handle outliers
    'outlier_method': 'cap',  # 'cap', 'winsorize', or 'remove'
    'apply_scaling': False,  # Usually done in model pipeline, not here
    'scaling_method': 'robust',  # 'standard', 'robust', 'minmax', 'quantile', 'power'
    'create_interactions': True,  # Create interaction features
    'max_interactions': 10,  # Maximum number of interaction features
    'create_bins': False,  # Create binned features (can increase dimensionality)
    'apply_log_transform': True,  # Apply log transform to skewed features
    'remove_multicollinear': True,  # Remove highly correlated features
    'remove_low_variance': True,  # Remove low variance features
    'feature_selection': None,  # 'f_classif', 'mutual_info_classif', or None
    'n_features_select': 50  # Number of features to select (if feature_selection is set)
}

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Create model subdirectories
for model_dir in MODEL_SUBDIRS.values():
    model_dir.mkdir(parents=True, exist_ok=True)


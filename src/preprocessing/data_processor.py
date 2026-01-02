"""
Optimized data preprocessing script
Processes raw diabetes dataset and saves processed version with validation and quality checks
"""
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import sys

try:
    from ..config import RAW_DATA_FILE, PROCESSED_DATA_FILE, MODELS_DIR
    from ..utils import preprocess_data
except ImportError:
    # Fallback for when running as script
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import RAW_DATA_FILE, PROCESSED_DATA_FILE, MODELS_DIR
    from utils import preprocess_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_raw_data(df: pd.DataFrame) -> dict:
    """
    Validate raw data quality
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'has_target': 'Target' in df.columns,
        'target_distribution': None
    }
    
    if 'Target' in df.columns:
        validation_results['target_distribution'] = df['Target'].value_counts().to_dict()
        validation_results['target_classes'] = df['Target'].nunique()
    
    return validation_results


def print_summary(df_raw: pd.DataFrame, df_processed: pd.DataFrame, validation: dict):
    """Print detailed processing summary"""
    print(f"\n{'='*60}")
    print("DATA PREPROCESSING SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n[Raw Data]")
    print(f"  Shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
    print(f"  Duplicate rows: {validation['duplicate_rows']}")
    print(f"  Missing values: {sum(validation['missing_values'].values())} total")
    
    if validation['has_target']:
        print(f"  Target classes: {validation['target_classes']}")
        print(f"  Target distribution:")
        for cls, count in list(validation['target_distribution'].items())[:5]:
            print(f"    - {cls}: {count}")
        if len(validation['target_distribution']) > 5:
            print(f"    ... and {len(validation['target_distribution']) - 5} more")
    
    print(f"\n[Processed Data]")
    print(f"  Shape: {df_processed.shape[0]} rows × {df_processed.shape[1]} columns")
    
    # Count feature types
    feature_cols = [c for c in df_processed.columns if c != 'Target']
    numerical_features = len([c for c in feature_cols if not any(x in c for x in ['_', 'x0_', 'x1_'])])
    categorical_features = len(feature_cols) - numerical_features
    
    print(f"  Numerical features: {numerical_features}")
    print(f"  Encoded categorical features: {categorical_features}")
    print(f"  Total features: {len(feature_cols)}")
    
    print(f"\n[Memory Usage]")
    print(f"  Raw data: {df_raw.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"  Processed data: {df_processed.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"{'='*60}\n")


def main():
    """Main function to process raw data with optimizations"""
    start_time = datetime.now()
    
    try:
        logger.info("="*60)
        logger.info("Starting optimized data preprocessing...")
        logger.info("="*60)
        
        # Load raw data with optimizations
        if not RAW_DATA_FILE.exists():
            raise FileNotFoundError(f"Raw data file not found: {RAW_DATA_FILE}")
        
        logger.info(f"Loading data from {RAW_DATA_FILE}")
        df = pd.read_csv(
            RAW_DATA_FILE,
            low_memory=False,  # Optimize memory usage
            na_values=['', 'NA', 'N/A', 'null', 'NULL', 'None']  # Standardize missing values
        )
        
        logger.info(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Validate data
        logger.info("Validating data quality...")
        validation = validate_raw_data(df)
        
        if validation['duplicate_rows'] > 0:
            logger.warning(f"Found {validation['duplicate_rows']} duplicate rows.")
        
        # Preprocess data with optimizations
        logger.info("Preprocessing data...")
        try:
            from ..config import USE_ADVANCED_PREPROCESSING, ADVANCED_PREPROCESSING_OPTIONS
        except ImportError:
            from config import USE_ADVANCED_PREPROCESSING, ADVANCED_PREPROCESSING_OPTIONS
        
        df_processed, encoders = preprocess_data(
            df, 
            fit_encoders=True,
            handle_missing='fill',  # Fill missing values in numerical columns
            validate_data=True,
            use_advanced=USE_ADVANCED_PREPROCESSING,
            advanced_options=ADVANCED_PREPROCESSING_OPTIONS if USE_ADVANCED_PREPROCESSING else None
        )
        
        # Save processed data
        logger.info(f"Saving processed data to {PROCESSED_DATA_FILE}")
        PROCESSED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        df_processed.to_csv(PROCESSED_DATA_FILE, index=False)
        logger.info("✓ Processed data saved successfully")
        
        # Save encoders for later use (in models root, shared across all models)
        MODELS_DIR.mkdir(exist_ok=True)
        encoders_path = MODELS_DIR / "data_encoders.pkl"
        joblib.dump(encoders, encoders_path, compress=3)  # Compress to save space
        logger.info(f"✓ Data encoders saved to {encoders_path}")
        
        # Calculate processing time
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"\n{'='*60}")
        logger.info("Data preprocessing completed successfully!")
        logger.info(f"Processing time: {duration:.2f} seconds")
        logger.info(f"{'='*60}")
        
        # Print detailed summary
        print_summary(df, df_processed, validation)
    
    except Exception as e:
        logger.error(f"Error during data preprocessing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()


"""
Utility functions for data processing, model training, and evaluation
"""
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

try:
    from .config import (
        RAW_DATA_FILE, PROCESSED_DATA_FILE, MODELS_DIR, RESULTS_DIR,
        RANDOM_STATE, TEST_SIZE, MODEL_NAMES
    )
except ImportError:
    # Fallback for when running as script
    from config import (
        RAW_DATA_FILE, PROCESSED_DATA_FILE, MODELS_DIR, RESULTS_DIR,
        RANDOM_STATE, TEST_SIZE, MODEL_NAMES
    )

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Cache for loaded data to avoid reloading
_data_cache: Dict[Path, pd.DataFrame] = {}
# Cache for processed and split data to share across models
_processed_data_cache: Optional[Tuple[pd.DataFrame, Dict[str, Any]]] = None
_split_data_cache: Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, LabelEncoder]] = None


def load_data(file_path: Path = None, use_cache: bool = True) -> pd.DataFrame:
    """
    Load dataset from CSV file with caching
    
    Args:
        file_path: Path to CSV file. If None, uses processed data file from config
        use_cache: Whether to use cached data if available
        
    Returns:
        DataFrame containing the dataset
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    if file_path is None:
        file_path = PROCESSED_DATA_FILE
    
    try:
        # Check cache first
        if use_cache and file_path in _data_cache:
            logger.debug(f"Using cached data for {file_path}")
            return _data_cache[file_path].copy()
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path, low_memory=False)
        
        if df.empty:
            raise ValueError(f"Data file is empty: {file_path}")
        
        # Optimize data types to reduce memory
        df = _optimize_dtypes(df)
        
        # Cache the data
        if use_cache:
            _data_cache[file_path] = df.copy()
        
        logger.info(f"Loaded data from {file_path}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame dtypes to reduce memory usage
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with optimized dtypes
    """
    df = df.copy()
    
    # Optimize integer columns
    for col in df.select_dtypes(include=['int64']).columns:
        col_min = df[col].min()
        col_max = df[col].max()
        
        if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
            df[col] = df[col].astype(np.int8)
        elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16)
        elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)
    
    # Optimize float columns
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Convert object columns to category if low cardinality
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'Target':  # Don't convert target
            num_unique = df[col].nunique()
            num_total = len(df)
            if num_unique / num_total < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
    
    return df


def _identify_column_types(df: pd.DataFrame, target_col: str = 'Target') -> Tuple[list, list]:
    """
    Efficiently identify numerical and categorical columns
    
    Args:
        df: Input DataFrame
        target_col: Name of target column to exclude
        
    Returns:
        Tuple of (numerical_columns, categorical_columns)
    """
    # Exclude target column
    feature_cols = [col for col in df.columns if col != target_col]
    
    if not feature_cols:
        return [], []
    
    # Vectorized approach: try to convert all columns at once
    numerical_columns = []
    categorical_columns = []
    
    # Get dtypes first (faster check)
    numeric_dtypes = ['int64', 'int32', 'float64', 'float32']
    
    for col in feature_cols:
        # Fast path: already numeric dtype
        if df[col].dtype in numeric_dtypes:
            numerical_columns.append(col)
            continue
        
        # Check if can be converted to numeric
        try:
            coerced = pd.to_numeric(df[col], errors='coerce')
            valid_ratio = coerced.notna().mean()
            
            if valid_ratio >= 0.95:  # Allow small amount of missing values
                numerical_columns.append(col)
            else:
                categorical_columns.append(col)
        except Exception:
            categorical_columns.append(col)
    
    return numerical_columns, categorical_columns


def preprocess_data(df: pd.DataFrame, fit_encoders: bool = True, 
                   encoders: Optional[Dict[str, Any]] = None,
                   handle_missing: str = 'ignore',
                   validate_data: bool = True,
                   use_advanced: bool = False,
                   advanced_options: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Optimized preprocessing: identify numerical/categorical columns and encode categorical features
    
    Args:
        df: Input DataFrame
        fit_encoders: If True, fit new encoders. If False, use provided encoders
        encoders: Dictionary containing fitted encoders (OneHotEncoder, LabelEncoder)
        handle_missing: How to handle missing values ('ignore', 'drop', 'fill')
        validate_data: Whether to validate data quality
        use_advanced: Whether to use advanced preprocessing techniques
        advanced_options: Dictionary of advanced preprocessing options
        
    Returns:
        Tuple of (processed DataFrame, encoders dictionary)
    """
    try:
        # Apply advanced preprocessing if requested
        if use_advanced:
            try:
                from .preprocessing.advanced_preprocessing import advanced_preprocessing_pipeline
            except ImportError:
                from preprocessing.advanced_preprocessing import advanced_preprocessing_pipeline
            
            if advanced_options is None:
                advanced_options = {}
            
            logger.info("Applying advanced preprocessing techniques...")
            df, advanced_artifacts = advanced_preprocessing_pipeline(
                df,
                target_col='Target',
                **advanced_options
            )
            
            # Merge advanced artifacts into encoders dict
            if encoders is None:
                encoders = {}
            encoders['advanced_artifacts'] = advanced_artifacts
        # Data validation
        if validate_data:
            if df.empty:
                raise ValueError("Input DataFrame is empty")
            
            if df.shape[0] < 10:
                logger.warning(f"Very small dataset: {df.shape[0]} rows")
            
            # Check for completely empty columns
            empty_cols = df.columns[df.isnull().all()].tolist()
            if empty_cols:
                logger.warning(f"Found empty columns: {empty_cols}. They will be dropped.")
                df = df.drop(columns=empty_cols)
        
        # Identify column types efficiently
        numerical_columns, category_columns = _identify_column_types(df)
        
        # Reduced logging for cleaner output
        if len(numerical_columns) > 0 or len(category_columns) > 0:
            logger.debug(f"Identified {len(numerical_columns)} numerical and {len(category_columns)} categorical columns")
        
        # Handle missing values in numerical columns
        if handle_missing != 'ignore' and numerical_columns:
            missing_numerical = df[numerical_columns].isnull().sum()
            if missing_numerical.sum() > 0:
                logger.debug(f"Found missing values in numerical columns: {missing_numerical[missing_numerical > 0].to_dict()}")
                if handle_missing == 'fill':
                    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())
                elif handle_missing == 'drop':
                    df = df.dropna(subset=numerical_columns)
        
        # Process categorical columns
        if category_columns:
            if fit_encoders:
                # Fit new encoders
                encoder = OneHotEncoder(
                    sparse_output=False, 
                    handle_unknown='ignore',
                    drop='if_binary'  # Optimize: drop one column for binary features
                )
                
                # Check for empty categorical columns
                non_empty_cats = [col for col in category_columns if df[col].notna().any()]
                if len(non_empty_cats) < len(category_columns):
                    logger.warning(f"Skipping {len(category_columns) - len(non_empty_cats)} empty categorical columns")
                
                if non_empty_cats:
                    encoded_df = encoder.fit_transform(df[non_empty_cats])
                    encoded_df = pd.DataFrame(
                        encoded_df, 
                        columns=encoder.get_feature_names_out(non_empty_cats),
                        index=df.index
                    )
                else:
                    encoded_df = pd.DataFrame(index=df.index)
                
                encoders = {
                    'onehot': encoder,
                    'categorical_columns': non_empty_cats,
                    'label': None  # Will be set separately for target
                }
            else:
                # Use existing encoders
                if encoders is None or 'onehot' not in encoders:
                    raise ValueError("Encoders must be provided when fit_encoders=False")
                
                encoder = encoders['onehot']
                cat_cols = encoders.get('categorical_columns', category_columns)
                
                # Handle missing categorical columns
                missing_cats = set(cat_cols) - set(df.columns)
                if missing_cats:
                    logger.warning(f"Missing categorical columns in input: {missing_cats}")
                    cat_cols = [col for col in cat_cols if col in df.columns]
                
                if cat_cols:
                    encoded_df = encoder.transform(df[cat_cols])
                    encoded_df = pd.DataFrame(
                        encoded_df,
                        columns=encoder.get_feature_names_out(cat_cols),
                        index=df.index
                    )
                else:
                    encoded_df = pd.DataFrame(index=df.index)
        else:
            encoded_df = pd.DataFrame(index=df.index)
            if fit_encoders:
                encoders = {
                    'onehot': None,
                    'categorical_columns': [],
                    'label': None
                }
        
        # Process numerical columns
        if numerical_columns:
            numerical_df = df[numerical_columns].copy()
            # Convert to float efficiently
            numerical_df = numerical_df.astype(float)
        else:
            numerical_df = pd.DataFrame(index=df.index)
        
        # Combine numerical and encoded categorical features efficiently
        if not numerical_df.empty and not encoded_df.empty:
            df_processed = pd.concat([numerical_df, encoded_df], axis=1)
        elif not numerical_df.empty:
            df_processed = numerical_df
        elif not encoded_df.empty:
            df_processed = encoded_df
        else:
            raise ValueError("No features to process")
        
        # Ensure all columns are numeric
        df_processed = df_processed.astype(float)
        
        # Add target column if present
        if 'Target' in df.columns:
            df_processed['Target'] = df['Target'].astype(str)  # Keep as string for now
        
        logger.debug(f"Preprocessed data: {df_processed.shape[0]} rows, {df_processed.shape[1]} columns")
        logger.debug(f"  - Numerical features: {len(numerical_columns)}")
        logger.debug(f"  - Encoded categorical features: {encoded_df.shape[1] if not encoded_df.empty else 0}")
        
        return df_processed, encoders
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise


def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """
    Prepare data for training: encode target and split features/labels
    
    Args:
        df: Processed DataFrame with Target column
        
    Returns:
        Tuple of (X features, y target, label_encoder)
    """
    try:
        le = LabelEncoder()
        df = df.copy()
        df['Target'] = le.fit_transform(df['Target'])
        
        X = df.drop(columns=['Target'])
        y = df['Target']
        
        logger.info(f"Prepared training data: X shape {X.shape}, y shape {y.shape}")
        logger.info(f"Target classes: {le.classes_}")
        
        return X, y, le
    
    except Exception as e:
        logger.error(f"Error preparing training data: {str(e)}")
        raise


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE, 
               random_state: int = RANDOM_STATE, use_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets with caching
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of test set
        random_state: Random seed
        use_cache: Whether to use cached split if available
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    global _split_data_cache
    
    try:
        # Check cache
        if use_cache and _split_data_cache is not None:
            X_train, X_test, y_train, y_test, _ = _split_data_cache
            logger.debug("Using cached split data")
            return X_train, X_test, y_train, y_test
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f"Split data: Train {X_train.shape[0]} samples, Test {X_test.shape[0]} samples")
        
        # Cache the split (without label_encoder for now)
        if use_cache:
            _split_data_cache = (X_train, X_test, y_train, y_test, None)
        
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logger.error(f"Error splitting data: {str(e)}")
        raise


def get_prepared_data(use_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, LabelEncoder, Dict[str, Any]]:
    """
    Get prepared training data with caching (shared across all models)
    
    Args:
        use_cache: Whether to use cached data if available
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, label_encoder, encoders)
    """
    global _processed_data_cache, _split_data_cache
    
    try:
        # Check if we have cached split data
        if use_cache and _split_data_cache is not None:
            X_train, X_test, y_train, y_test, label_encoder = _split_data_cache
            if label_encoder is not None:
                # Get encoders from processed data cache
                _, encoders = _processed_data_cache if _processed_data_cache else (None, None)
                logger.debug("Using cached prepared data")
                return X_train, X_test, y_train, y_test, label_encoder, encoders
        
        # Load and preprocess data
        df = load_data(use_cache=use_cache)
        df_processed, encoders = preprocess_data(df, fit_encoders=True)
        
        # Cache processed data
        if use_cache:
            _processed_data_cache = (df_processed, encoders)
        
        # Prepare training data
        X, y, label_encoder = prepare_training_data(df_processed)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y, use_cache=use_cache)
        
        # Keep as DataFrame/Series for sklearn pipeline compatibility (needs feature names)
        # Cache split data with label encoder
        if use_cache:
            _split_data_cache = (X_train, X_test, y_train, y_test, label_encoder)
        
        logger.debug(f"Prepared data for training: X_train {X_train.shape}, X_test {X_test.shape}")
        return X_train, X_test, y_train, y_test, label_encoder, encoders
    
    except Exception as e:
        logger.error(f"Error getting prepared data: {str(e)}")
        raise


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                  model_name: str, save_plots: bool = True, label_encoder: Optional[LabelEncoder] = None) -> Dict[str, Any]:
    """
    Evaluate model and generate comprehensive metrics
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model (for saving files)
        save_plots: Whether to save evaluation plots
        label_encoder: Optional label encoder to decode class names
        
    Returns:
        Dictionary containing evaluation metrics
    """
    try:
        y_pred = model.predict(X_test)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate additional metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        precision_micro = precision_score(y_test, y_pred, average='micro', zero_division=0)
        precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        recall_micro = recall_score(y_test, y_pred, average='micro', zero_division=0)
        recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Print summary metrics (detailed report saved to log file)
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f} | F1-Macro: {f1_macro:.4f} | F1-Weighted: {f1_weighted:.4f}")
        logger.debug(f"\n{model_name} - Detailed Evaluation Metrics:")
        logger.debug(f"  Accuracy: {accuracy:.4f}")
        logger.debug(f"  Precision (macro/micro/weighted): {precision_macro:.4f}/{precision_micro:.4f}/{precision_weighted:.4f}")
        logger.debug(f"  Recall (macro/micro/weighted): {recall_macro:.4f}/{recall_micro:.4f}/{recall_weighted:.4f}")
        logger.debug(f"  F1-Score (macro/micro/weighted): {f1_macro:.4f}/{f1_micro:.4f}/{f1_weighted:.4f}")
        logger.debug(f"\n{model_name} Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
        
        # Save confusion matrix plot (heatmap)
        if save_plots:
            try:
                # Get class labels if available
                if label_encoder is not None:
                    class_names = label_encoder.classes_
                else:
                    # Try to get unique classes from y_test
                    class_names = sorted(y_test.unique())
                
                # Create figure with two subplots: raw counts and normalized
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
                
                # Plot 1: Raw confusion matrix
                disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
                disp1.plot(ax=ax1, cmap='Blues', values_format='d')
                ax1.set_title(f'{model_name} - Confusion Matrix (Counts)\nAccuracy: {accuracy:.4f} | F1-Macro: {f1_macro:.4f}', 
                            fontsize=12, fontweight='bold')
                ax1.set_xlabel('Predicted', fontsize=11, fontweight='bold')
                ax1.set_ylabel('Actual', fontsize=11, fontweight='bold')
                
                # Plot 2: Normalized confusion matrix (percentages)
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
                disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
                disp2.plot(ax=ax2, cmap='Oranges', values_format='.2%')
                ax2.set_title(f'{model_name} - Confusion Matrix (Normalized)\nPercentage of Actual Class', 
                            fontsize=12, fontweight='bold')
                ax2.set_xlabel('Predicted', fontsize=11, fontweight='bold')
                ax2.set_ylabel('Actual', fontsize=11, fontweight='bold')
                
                plt.tight_layout()
                
                # Save to results directory
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                plot_path = RESULTS_DIR / f"{model_name.lower().replace(' ', '_')}_heatmap.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                logger.info(f"✓ Saved confusion matrix heatmap to {plot_path}")
            except Exception as e:
                logger.warning(f"Could not save confusion matrix plot: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())
        
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'accuracy': accuracy,
            'precision': {
                'macro': precision_macro,
                'micro': precision_micro,
                'weighted': precision_weighted
            },
            'recall': {
                'macro': recall_macro,
                'micro': recall_micro,
                'weighted': recall_weighted
            },
            'f1_score': {
                'macro': f1_macro,
                'micro': f1_micro,
                'weighted': f1_weighted
            }
        }
    
    except Exception as e:
        logger.error(f"Error evaluating model {model_name}: {str(e)}")
        raise


def save_model(model: Any, model_name: str, label_encoder: Optional[LabelEncoder] = None,
              encoders: Optional[Dict[str, Any]] = None) -> Path:
    """
    Save trained model and associated encoders in organized directory structure
    
    Args:
        model: Trained model
        model_name: Name of the model (key in MODEL_NAMES)
        label_encoder: Label encoder for target variable
        encoders: Dictionary of feature encoders
        
    Returns:
        Path to saved model file
    """
    try:
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(MODEL_NAMES.keys())}")
        
        # Get model-specific directory
        try:
            from .config import MODEL_SUBDIRS
        except ImportError:
            from config import MODEL_SUBDIRS
        
        model_dir = MODEL_SUBDIRS.get(model_name, MODELS_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model in its own directory
        model_path = model_dir / MODEL_NAMES[model_name]
        joblib.dump(model, model_path)
        logger.info(f"✓ Saved model to {model_path}")
        
        # Save label encoder if provided
        if label_encoder is not None:
            encoder_path = model_dir / f"{model_name}_label_encoder.pkl"
            joblib.dump(label_encoder, encoder_path)
            logger.info(f"✓ Saved label encoder to {encoder_path}")
        
        # Save feature encoders if provided
        if encoders is not None:
            encoders_path = model_dir / f"{model_name}_encoders.pkl"
            joblib.dump(encoders, encoders_path)
            logger.info(f"✓ Saved feature encoders to {encoders_path}")
        
        return model_path
    
    except Exception as e:
        logger.error(f"Error saving model {model_name}: {str(e)}")
        raise


def load_model(model_name: str, load_encoders: bool = True) -> Tuple[Any, Optional[LabelEncoder], Optional[Dict[str, Any]]]:
    """
    Load trained model and associated encoders from organized directory structure
    
    Args:
        model_name: Name of the model (key in MODEL_NAMES)
        load_encoders: Whether to load encoders
        
    Returns:
        Tuple of (model, label_encoder, encoders)
    """
    try:
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(MODEL_NAMES.keys())}")
        
        # Get model-specific directory
        try:
            from .config import MODEL_SUBDIRS
        except ImportError:
            from config import MODEL_SUBDIRS
        
        model_dir = MODEL_SUBDIRS.get(model_name, MODELS_DIR)
        model_path = model_dir / MODEL_NAMES[model_name]
        
        # Fallback to old location if not found in new structure
        if not model_path.exists():
            old_path = MODELS_DIR / MODEL_NAMES[model_name]
            if old_path.exists():
                logger.warning(f"Model not found in new structure, loading from old location: {old_path}")
                model_path = old_path
                model_dir = MODELS_DIR
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        logger.info(f"✓ Loaded model from {model_path}")
        
        label_encoder = None
        encoders = None
        
        if load_encoders:
            # Try to load label encoder from model directory
            encoder_path = model_dir / f"{model_name}_label_encoder.pkl"
            if not encoder_path.exists():
                # Fallback to old location
                encoder_path = MODELS_DIR / f"{model_name}_label_encoder.pkl"
            
            if encoder_path.exists():
                label_encoder = joblib.load(encoder_path)
                logger.info(f"✓ Loaded label encoder from {encoder_path}")
            
            # Try to load feature encoders from model directory
            encoders_path = model_dir / f"{model_name}_encoders.pkl"
            if not encoders_path.exists():
                # Fallback to old location
                encoders_path = MODELS_DIR / f"{model_name}_encoders.pkl"
            
            if encoders_path.exists():
                encoders = joblib.load(encoders_path)
                logger.info(f"✓ Loaded feature encoders from {encoders_path}")
        
        return model, label_encoder, encoders
    
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        raise

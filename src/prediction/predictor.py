"""
Prediction module for diabetes assessment
Uses ensemble of 5 trained models with weighted voting
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from ..config import MODEL_NAMES, MODEL_WEIGHTS
    from ..utils import load_model, preprocess_data
except ImportError:
    # Fallback for when running as script
    from config import MODEL_NAMES, MODEL_WEIGHTS
    from utils import load_model, preprocess_data

logger = logging.getLogger(__name__)

# Cache loaded models
_models_cache: Dict[str, Any] = {}
_label_encoders_cache: Dict[str, Any] = {}
_encoders_cache: Dict[str, Any] = {}
_model_columns_cache: Dict[str, list] = {}  # Cache feature columns for each model


def _load_all_models():
    """Load all models and encoders into cache (lazy loading)"""
    if _models_cache:
        return  # Already loaded
    
    model_names = ['knn', 'logistic', 'randomforest', 'svm', 'xgboost']
    
    for model_name in model_names:
        try:
            model, label_encoder, encoders = load_model(model_name, load_encoders=True)
            _models_cache[model_name] = model
            _label_encoders_cache[model_name] = label_encoder
            _encoders_cache[model_name] = encoders
            logger.info(f"Loaded {model_name} model and encoders")
        except Exception as e:
            logger.error(f"Error loading {model_name} model: {str(e)}")
            raise


def _get_target_classes() -> list:
    """
    Get all possible target classes from label encoder
    Uses XGBoost model's label encoder as reference
    """
    try:
        if 'xgboost' not in _label_encoders_cache:
            _load_all_models()
        
        label_encoder = _label_encoders_cache['xgboost']
        if label_encoder is not None:
            return label_encoder.classes_.tolist()
        else:
            # Fallback: try to get from model
            model = _models_cache['xgboost']
            # This is a workaround if label encoder is not saved
            logger.warning("Label encoder not found, using default classes")
            return []
    except Exception as e:
        logger.error(f"Error getting target classes: {str(e)}")
        return []


def prediction(user_data: pd.DataFrame, return_details: bool = False) -> str:
    """
    Predict diabetes type using ensemble of 5 models
    
    Args:
        user_data: DataFrame containing user input data (single row)
                   Must contain all required features except 'Target'
    
    Returns:
        If return_details=False: Predicted diabetes type as string
        If return_details=True: Dictionary with prediction and details
    
    Raises:
        ValueError: If user_data is invalid or missing required features
        RuntimeError: If models cannot be loaded or prediction fails
    """
    try:
        # Validate input
        if user_data.empty:
            raise ValueError("User data is empty")
        
        if user_data.shape[0] > 1:
            logger.warning(f"User data has {user_data.shape[0]} rows, using first row only")
            user_data = user_data.iloc[[0]]
        
        # Load models if not already loaded
        if not _models_cache:
            _load_all_models()
        
        # Get reference encoders (use XGBoost as reference since it's most reliable)
        if 'xgboost' not in _encoders_cache:
            raise RuntimeError("XGBoost model and encoders not found")
        
        reference_encoders = _encoders_cache['xgboost']
        reference_label_encoder = _label_encoders_cache['xgboost']
        
        if reference_encoders is None or reference_label_encoder is None:
            raise RuntimeError("Required encoders not found. Please retrain models with save_encoders=True")
        
        # Preprocess user data using saved encoders
        user_data_processed, _ = preprocess_data(
            user_data,
            fit_encoders=False,
            encoders=reference_encoders
        )
        
        # Remove Target column if present in user input (prediction input does not include target variable)
        if 'Target' in user_data_processed.columns:
            user_data_processed = user_data_processed.drop(columns=['Target'])
        
        # Get feature columns from reference model (cached)
        if 'xgboost' not in _model_columns_cache:
            reference_model = _models_cache['xgboost']
            try:
                # Try to get feature names from pipeline (sklearn Pipeline stores feature names)
                if hasattr(reference_model, 'feature_names_in_'):
                    model_columns = reference_model.feature_names_in_
                elif hasattr(reference_model, 'named_steps'):
                    # Try to get from classifier step in pipeline
                    classifier = reference_model.named_steps.get('classifier')
                    if classifier and hasattr(classifier, 'feature_names_in_'):
                        model_columns = classifier.feature_names_in_
                    else:
                        # Use input data columns as fallback
                        model_columns = user_data_processed.columns.tolist()
                        logger.warning("Could not get feature names from classifier, using processed columns")
                else:
                    # Last resort: use all columns
                    model_columns = user_data_processed.columns.tolist()
                    logger.warning("Could not get feature names from model, using all processed columns")
            except Exception as e:
                logger.warning(f"Error getting feature names: {str(e)}, using processed columns")
                model_columns = user_data_processed.columns.tolist()
            _model_columns_cache['xgboost'] = model_columns
        else:
            model_columns = _model_columns_cache['xgboost']
        
        # Optimized column alignment using vectorized operations
        user_cols = set(user_data_processed.columns)
        model_cols_set = set(model_columns)
        
        missing_cols = model_cols_set - user_cols
        extra_cols = user_cols - model_cols_set
        
        if missing_cols:
            logger.warning(f"Missing columns: {len(missing_cols)}. Filling with zeros.")
            # Vectorized: create DataFrame with zeros for missing columns
            missing_df = pd.DataFrame(0, index=user_data_processed.index, columns=list(missing_cols))
            user_data_processed = pd.concat([user_data_processed, missing_df], axis=1)
        
        if extra_cols:
            logger.warning(f"Extra columns: {len(extra_cols)}. Dropping them.")
            user_data_processed = user_data_processed.drop(columns=list(extra_cols))
        
        # Ensure correct column order (vectorized)
        user_data_processed = user_data_processed.reindex(columns=model_columns, fill_value=0)
        
        # Get all possible target classes for voting
        target_classes = _get_target_classes()
        if not target_classes:
            # Fallback: use classes from reference label encoder
            target_classes = reference_label_encoder.classes_.tolist()
        
        find_result = {str(target): 0 for target in target_classes}
        
        # Get predictions from all models (optimized batch processing)
        predictions = {}
        model_names = ['knn', 'logistic', 'randomforest', 'svm', 'xgboost']
        
        # Batch predict for efficiency (if all models support it)
        for model_name in model_names:
            try:
                model = _models_cache[model_name]
                label_encoder = _label_encoders_cache[model_name]
                
                if model is None:
                    logger.warning(f"{model_name} model not available, skipping")
                    continue
                
                # Predict (Pipeline accepts DataFrame directly)
                pred_encoded = model.predict(user_data_processed)
                
                # Decode prediction (handle both single value and array)
                if isinstance(pred_encoded, np.ndarray) and len(pred_encoded) > 0:
                    pred_value = pred_encoded[0]
                else:
                    pred_value = pred_encoded
                
                if label_encoder is not None:
                    pred_decoded = label_encoder.inverse_transform([pred_value])[0]
                else:
                    # Fallback: use reference label encoder
                    pred_decoded = reference_label_encoder.inverse_transform([pred_value])[0]
                
                predictions[model_name] = str(pred_decoded)
                
                # Add weighted vote
                weight = MODEL_WEIGHTS.get(model_name, 50)
                find_result[predictions[model_name]] += weight
                
                logger.debug(f"{model_name} prediction: {predictions[model_name]} (weight: {weight})")
            
            except Exception as e:
                logger.error(f"Error getting prediction from {model_name}: {str(e)}")
                continue
        
        if not predictions:
            raise RuntimeError("No models could make predictions")
        
        # Get final result (highest weighted vote)
        result = max(find_result, key=find_result.get)
        total_votes = sum(find_result.values())
        confidence = (find_result[result] / total_votes * 100) if total_votes > 0 else 0
        
        logger.info(f"Final prediction: {result}")
        logger.info(f"Confidence: {confidence:.2f}%")
        logger.info(f"Voting results: {find_result}")
        logger.info(f"Individual predictions: {predictions}")
        
        if return_details:
            return {
                'prediction': result,
                'confidence': confidence,
                'voting_results': find_result,
                'individual_predictions': predictions,
                'models_used': list(predictions.keys())
            }
        
        return result
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise


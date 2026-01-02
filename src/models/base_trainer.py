"""
Base trainer class for all ML models
Uses Optuna for hyperparameter optimization instead of GridSearchCV
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.trial import TrialState
import warnings
from sklearn.exceptions import ConvergenceWarning
from datetime import datetime
import uuid

# Import centralized logging
try:
    from ..logging_config import get_logger, setup_logging
except ImportError:
    from logging_config import get_logger, setup_logging

warnings.filterwarnings('ignore', message='.*experimental.*', module='optuna')
warnings.filterwarnings('ignore', category=UserWarning, module='optuna._experimental')
warnings.filterwarnings('ignore', message='.*Features.*are constant.*', module='sklearn.feature_selection')
warnings.filterwarnings('ignore', message='.*invalid value encountered in divide.*', module='sklearn.feature_selection')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn.feature_selection')
warnings.filterwarnings('ignore', message=".*'n_jobs'.*does not have any effect.*'liblinear'.*", module='sklearn.linear_model')

try:
    from ..config import (
        RANDOM_STATE, CV_FOLDS, SCORING_METRIC,
        N_TRIALS, OPTUNA_TIMEOUT, MODEL_TRIALS
    )
    from .. import config as config_module
    from ..utils import (
        get_prepared_data, evaluate_model, save_model
    )
except ImportError:
    # Fallback for when running as script
    from config import (
        RANDOM_STATE, CV_FOLDS, SCORING_METRIC,
        N_TRIALS, OPTUNA_TIMEOUT, MODEL_TRIALS
    )
    import config as config_module
    from utils import (
        get_prepared_data, evaluate_model, save_model
    )

logger = get_logger(__name__)


class BaseModelTrainer(ABC):
    """Base class for training ML models"""
    
    def __init__(self, model_name: str, use_scaler: bool = True):
        """
        Initialize base trainer
        
        Args:
            model_name: Name of the model (key in MODEL_NAMES config)
            use_scaler: Whether to use StandardScaler in pipeline
        """
        self.model_name = model_name
        self.use_scaler = use_scaler
        self.pipeline = None
        self.best_model = None
        self.label_encoder = None
        self.encoders = None
    
    @abstractmethod
    def get_classifier(self, params: Optional[Dict[str, Any]] = None):
        """Return the classifier instance with optional parameters"""
        pass
    
    @abstractmethod
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters using Optuna trial
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameters for the classifier
        """
        pass
    
    def create_pipeline(self, classifier_params: Optional[Dict[str, Any]] = None) -> Pipeline:
        """
        Create sklearn Pipeline with scaler and classifier.
        
        Args:
            classifier_params: Optional parameters with 'classifier__' prefix for nested parameters
            
        Returns:
            Configured Pipeline instance
        """
        steps = []
        
        if self.use_scaler:
            steps.append(('scaler', StandardScaler()))
        
        steps.append(('classifier', self.get_classifier()))
        
        pipeline = Pipeline(steps)
        
        if classifier_params:
            classifier_only_params = {k: v for k, v in classifier_params.items() 
                                     if k.startswith('classifier__')}
            if classifier_only_params:
                pipeline.set_params(**classifier_only_params)
        
        return pipeline
    
    def _objective(self, trial: optuna.Trial, X_train, y_train) -> float:
        """
        Objective function for Optuna hyperparameter optimization.
        
        Performs stratified cross-validation with intermediate value reporting
        for pruning. Handles both pandas DataFrames and numpy arrays.
        
        Args:
            trial: Optuna trial object
            X_train: Training features (numpy array or DataFrame)
            y_train: Training target (numpy array or Series)
            
        Returns:
            Mean CV score (precision_macro) across all folds
            
        Raises:
            optuna.TrialPruned: If trial should be pruned
            RuntimeError: If trial fails irrecoverably
        """
        try:
            # Get hyperparameters from trial
            params = self.suggest_hyperparameters(trial)
            
            # Create pipeline with suggested parameters
            pipeline = self.create_pipeline(params)
            
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            
            scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                try:
                    if isinstance(X_train, pd.DataFrame):
                        X_train_fold = X_train.iloc[train_idx]
                        X_val_fold = X_train.iloc[val_idx]
                        y_train_fold = y_train.iloc[train_idx]
                        y_val_fold = y_train.iloc[val_idx]
                    else:
                        X_train_fold = X_train[train_idx]
                        X_val_fold = X_train[val_idx]
                        y_train_fold = y_train[train_idx]
                        y_val_fold = y_train[val_idx]
                    
                    if isinstance(X_train_fold, pd.DataFrame):
                        if X_train_fold.empty or len(y_train_fold) == 0:
                            logger.debug(f"Trial {trial.number}: Empty fold {fold_idx}, skipping")
                            continue
                    else:
                        if len(X_train_fold) == 0 or len(y_train_fold) == 0:
                            logger.debug(f"Trial {trial.number}: Empty fold {fold_idx}, skipping")
                            continue
                    
                    # Fit pipeline with comprehensive error handling
                    try:
                        pipeline.fit(X_train_fold, y_train_fold)
                    except (ValueError, TypeError, AttributeError) as e:
                        # Silently skip common errors (e.g., negative values for MultinomialNB)
                        # Only log at debug level to reduce noise
                        logger.debug(f"Trial {trial.number}: Pipeline fit error in fold {fold_idx}: {str(e)}")
                        continue
                    except MemoryError as e:
                        logger.error(f"Trial {trial.number}: MemoryError during fit in fold {fold_idx}: {str(e)}")
                        raise RuntimeError(f"MemoryError: {str(e)}")
                    except Exception as e:
                        logger.debug(f"Trial {trial.number}: Unexpected fit error in fold {fold_idx}: {str(e)}")
                        continue
                    
                    # Predict with error handling
                    try:
                        val_pred = pipeline.predict(X_val_fold)
                    except (ValueError, AttributeError, RuntimeError) as e:
                        logger.debug(f"Trial {trial.number}: Prediction error in fold {fold_idx}: {str(e)}")
                        continue
                    except Exception as e:
                        logger.debug(f"Trial {trial.number}: Unexpected prediction error in fold {fold_idx}: {str(e)}")
                        continue
                    
                    if val_pred is None or len(val_pred) == 0:
                        logger.debug(f"Trial {trial.number}: Empty predictions for fold {fold_idx}")
                        continue
                    
                    # Calculate score with error handling
                    from sklearn.metrics import precision_score
                    try:
                        score = precision_score(y_val_fold, val_pred, average='macro', zero_division=0)
                    except Exception as e:
                        logger.debug(f"Trial {trial.number}: Score calculation error in fold {fold_idx}: {str(e)}")
                        score = 0.0
                    
                    if not np.isfinite(score):
                        logger.debug(f"Trial {trial.number}: Invalid score {score} for fold {fold_idx}, using 0.0")
                        score = 0.0
                    
                    scores.append(score)
                    trial.report(score, fold_idx)
                    
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
                except optuna.TrialPruned:
                    raise
                except MemoryError as e:
                    logger.error(f"Trial {trial.number}: MemoryError in fold {fold_idx}: {str(e)}")
                    raise RuntimeError(f"MemoryError: {str(e)}")
                except Exception as e:
                    logger.debug(f"Trial {trial.number}: Unexpected error in fold {fold_idx}: {str(e)}")
                    continue
            
            if not scores or len(scores) == 0:
                logger.debug(f"Trial {trial.number}: No valid scores computed - all folds failed, returning 0.0")
                return 0.0
            
            final_score = np.mean(scores)
            
            if not np.isfinite(final_score):
                logger.debug(f"Trial {trial.number}: Invalid final score {final_score}, returning 0.0")
                return 0.0
            
            return final_score
        
        except optuna.TrialPruned:
            raise
        except MemoryError as e:
            logger.error(f"Trial {trial.number}: MemoryError: {str(e)}")
            raise RuntimeError(f"MemoryError: {str(e)}")
        except Exception as e:
            logger.debug(f"Trial {trial.number}: Error in objective: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return 0.0
    
    def train(self, save_model_flag: bool = True, save_plots: bool = True, 
              n_trials: Optional[int] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Complete training pipeline: load data, optimize hyperparameters with Optuna, 
        train model, evaluate, and save
        
        Args:
            save_model_flag: Whether to save the trained model
            save_plots: Whether to save evaluation plots
            n_trials: Number of optimization trials (overrides config if provided)
            timeout: Timeout in seconds (overrides config if provided)
            
        Returns:
            Dictionary containing training results and metrics
        """
        try:
            root_logger = logging.getLogger()
            has_file_handler = any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)
            
            if not has_file_handler:
                from pathlib import Path
                project_root = Path(__file__).parent.parent.parent
                log_file = project_root / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                setup_logging(
                    log_file=log_file,
                    log_to_console=True,
                    log_to_file=True
                )
                logger.info(f"Setup file logging: {log_file}")
            
            warnings.filterwarnings('ignore', category=UserWarning, module='optuna')
            warnings.filterwarnings('ignore', message='.*experimental.*', module='optuna')
            warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')
            warnings.filterwarnings('ignore', message='.*Setting penalty=None will ignore.*', module='sklearn.linear_model')
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {self.model_name.upper()} model")
            logger.info(f"{'='*60}")
            
            # Load preprocessed training data with in-memory caching
            X_train, X_test, y_train, y_test, self.label_encoder, self.encoders = get_prepared_data(use_cache=True)
            
            if n_trials is None:
                n_trials = MODEL_TRIALS.get(self.model_name, N_TRIALS)
            timeout = timeout or OPTUNA_TIMEOUT
            
            optuna_logger = logging.getLogger('optuna')
            optuna_logger.setLevel(logging.WARNING)
            
            if self.model_name == 'knn':
                pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=CV_FOLDS, interval_steps=1)
            else:
                pruner = MedianPruner(n_startup_trials=8, n_warmup_steps=CV_FOLDS, interval_steps=1)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            unique_study_name = f"{self.model_name}_optimization_{timestamp}_{unique_id}"
            
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(
                    seed=RANDOM_STATE, 
                    n_startup_trials=10, 
                    multivariate=True, 
                    constant_liar=True,
                    warn_independent_sampling=False
                ),
                pruner=pruner,
                study_name=unique_study_name,
                load_if_exists=False,
                storage=None
            )
            
            logger.info(f"Created new Optuna study: {unique_study_name} (trials will start from 0)")
            
            logger.info(f"Starting Optuna optimization: {n_trials} trials for {self.model_name}")
            
            # Optimize hyperparameters with progress bar
            # Use config_module.OPTUNA_N_JOBS to get current value (allows runtime modification)
            study.optimize(
                lambda trial: self._objective(trial, X_train, y_train),
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=config_module.OPTUNA_N_JOBS,
                show_progress_bar=True  # Show progress bar
            )
            
            optuna_logger.setLevel(logging.INFO)
            
            completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
            if not completed_trials:
                logger.error(f"No trials completed successfully for {self.model_name}. All trials failed or were pruned.")
                raise RuntimeError(f"No trials completed successfully for {self.model_name}. All trials failed or were pruned.")
            
            logger.info(f"✓ Optimization completed - Best CV score: {study.best_value:.4f}")
            
            logger.info("Training final model...")
            best_params = study.best_params
            base_pipeline = self.create_pipeline(best_params)
            
            if hasattr(base_pipeline, '_use_calibration') and base_pipeline._use_calibration:
                from sklearn.calibration import CalibratedClassifierCV
                calibration_method = getattr(base_pipeline, '_calibration_method', 'isotonic')
                self.best_model = CalibratedClassifierCV(
                    base_pipeline,
                    method=calibration_method,
                    cv=3
                )
            else:
                self.best_model = base_pipeline
            
            self.best_model.fit(X_train, y_train)
            
            if hasattr(self.best_model, 'base_estimator'):
                self.best_model.base_estimator.best_params_ = best_params
                self.best_model.base_estimator.best_score_ = study.best_value
                self.best_model.best_params_ = best_params
                self.best_model.best_score_ = study.best_value
            else:
                self.best_model.best_params_ = best_params
                self.best_model.best_score_ = study.best_value
            
            logger.info("Evaluating model...")
            evaluation_results = evaluate_model(
                self.best_model,
                X_test,
                y_test,
                self.model_name,
                save_plots=save_plots,
                label_encoder=self.label_encoder
            )
            
            if save_model_flag:
                save_model(
                    self.best_model,
                    self.model_name,
                    self.label_encoder,
                    self.encoders
                )
            
            logger.info(f"✓ {self.model_name.upper()} training completed")
            
            return {
                'best_params': best_params,
                'best_score': study.best_value,
                'study': study,
                'evaluation': evaluation_results
            }
        
        except Exception as e:
            logger.error(f"Error during training {self.model_name}: {str(e)}")
            raise


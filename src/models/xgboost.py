"""
XGBoost Model Training Script
"""
import optuna
import numpy as np
from typing import Optional, Dict, Any
from xgboost import XGBClassifier
from .base_trainer import BaseModelTrainer

# Import centralized logging
try:
    from ..logging_config import get_logger, setup_logging
except ImportError:
    from logging_config import get_logger, setup_logging

logger = get_logger(__name__)


class XGBoostTrainer(BaseModelTrainer):
    """Trainer for XGBoost model"""
    
    def __init__(self):
        super().__init__(model_name='xgboost', use_scaler=False)  # Tree-based models don't need scaling
    
    def get_classifier(self, params: dict = None, num_classes: Optional[int] = None):
        """Return XGBoost classifier without early stopping (not compatible with Pipeline CV)"""
        # Dynamic num_class detection
        num_class = num_classes if num_classes is not None else 13
        
        # Note: early_stopping_rounds requires eval_set parameter in fit()
        # Pipeline.fit() doesn't support eval_set, so early stopping is disabled
        # Early stopping can be enabled in final model training if needed
        return XGBClassifier(
            objective='multi:softmax',
            num_class=num_class,
            use_label_encoder=False,
            eval_metric='mlogloss',
            # early_stopping_rounds removed - not compatible with Pipeline CV
            verbosity=0
        )
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict:
        """
        Hyperparameter search space for XGBoost
        
        Returns:
            Dictionary of hyperparameters with 'classifier__' prefix
        """
        n_estimators = trial.suggest_int('n_estimators', 200, 1000, step=50)
        max_depth = trial.suggest_int('max_depth', 3, 9, step=1)
        learning_rate = trial.suggest_float('learning_rate', 0.05, 0.3, log=True)
        
        subsample = trial.suggest_float('subsample', 0.6, 0.9, step=0.05)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 0.9, step=0.05)
        colsample_bylevel = trial.suggest_float('colsample_bylevel', 0.6, 0.9, step=0.05)
        colsample_bynode = trial.suggest_float('colsample_bynode', 0.6, 0.9, step=0.05)
        
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10, step=1)
        gamma = trial.suggest_float('gamma', 0, 5, step=0.5)
        
        reg_alpha = trial.suggest_float('reg_alpha', 0, 10, step=0.5)
        reg_lambda = trial.suggest_float('reg_lambda', 0, 10, step=0.5)
        
        use_monotonic_constraints = trial.suggest_categorical('use_monotonic_constraints', [False])
        tree_method = trial.suggest_categorical('tree_method', ['hist'])
        booster = trial.suggest_categorical('booster', ['gbtree'])
        
        try:
            from ..config import N_JOBS
        except (ImportError, ValueError):
            from config import N_JOBS
        
        params = {
            'classifier__n_estimators': n_estimators,
            'classifier__max_depth': max_depth,
            'classifier__learning_rate': learning_rate,
            'classifier__subsample': subsample,
            'classifier__colsample_bytree': colsample_bytree,
            'classifier__colsample_bylevel': colsample_bylevel,
            'classifier__colsample_bynode': colsample_bynode,
            'classifier__min_child_weight': min_child_weight,
            'classifier__gamma': gamma,
            'classifier__reg_alpha': reg_alpha,
            'classifier__reg_lambda': reg_lambda,
            'classifier__tree_method': tree_method,
            'classifier__booster': booster,
            'classifier__random_state': 42,
            'classifier__n_jobs': N_JOBS,
            'use_monotonic_constraints': use_monotonic_constraints
        }
        
        # DART-specific parameters
        if booster == 'dart':
            sample_type = trial.suggest_categorical('sample_type', ['uniform', 'uniform', 'weighted'])
            normalize_type = trial.suggest_categorical('normalize_type', ['tree', 'tree', 'forest'])
            rate_drop = trial.suggest_float('rate_drop', 0.1, 0.4, step=0.1)
            skip_drop = trial.suggest_float('skip_drop', 0.2, 0.5, step=0.1)
            
            params['classifier__sample_type'] = sample_type
            params['classifier__normalize_type'] = normalize_type
            params['classifier__rate_drop'] = rate_drop
            params['classifier__skip_drop'] = skip_drop
        
        return params
    
    def create_pipeline(self, classifier_params: Optional[Dict[str, Any]] = None):
        """
        Create pipeline with dynamic num_class and optional monotonic constraints
        """
        from sklearn.pipeline import Pipeline
        
        # Copy params to avoid side effects
        if classifier_params:
            params_copy = classifier_params.copy()
        else:
            params_copy = {}
        
        steps = []
        
        # XGBoost doesn't need scaler (tree-based)
        # Get classifier with dynamic num_class (will be set during training)
        classifier = self.get_classifier()
        steps.append(('classifier', classifier))
        
        pipeline = Pipeline(steps)
        
        # Set classifier parameters
        if params_copy:
            classifier_only_params = {k: v for k, v in params_copy.items() 
                                    if k.startswith('classifier__')}
            
            # Filter out DART-specific parameters if booster is not dart
            valid_params = {}
            booster_value = params_copy.get('classifier__booster', 'gbtree')
            dart_params = ['classifier__sample_type', 'classifier__normalize_type', 
                          'classifier__rate_drop', 'classifier__skip_drop']
            
            for key, value in classifier_only_params.items():
                # DART params only valid with dart booster
                if key in dart_params and booster_value != 'dart':
                    continue
                valid_params[key] = value
            
            if valid_params:
                try:
                    pipeline.set_params(**valid_params)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error setting classifier parameters: {str(e)}, trying without problematic params")
                    # Fallback: remove potentially problematic params
                    safe_params = {k: v for k, v in valid_params.items() 
                                 if k not in dart_params}
                    if safe_params:
                        try:
                            pipeline.set_params(**safe_params)
                        except Exception:
                            logger.warning(f"Failed to set even safe parameters, using defaults")
            
            # Configure monotonic constraints if enabled
            # Feature order determined during training phase
            use_monotonic = params_copy.get('use_monotonic_constraints', False)
            if use_monotonic:
                pipeline._use_monotonic_constraints = True
            else:
                pipeline._use_monotonic_constraints = False
        
        return pipeline
    
    def train(self, save_model_flag: bool = True, save_plots: bool = True, 
              n_trials: Optional[int] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Override train to handle dynamic num_class detection and monotonic constraints
        """
        # Handle relative imports with fallback for parallel execution
        try:
            from ..utils import get_prepared_data, evaluate_model, save_model
            from ..config import RANDOM_STATE, CV_FOLDS, N_TRIALS, OPTUNA_TIMEOUT, OPTUNA_N_JOBS
        except ImportError:
            # Fallback for when running as script or in parallel mode
            from utils import get_prepared_data, evaluate_model, save_model
            from config import RANDOM_STATE, CV_FOLDS, N_TRIALS, OPTUNA_TIMEOUT, OPTUNA_N_JOBS
        
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner
        import pandas as pd
        
        # Logger đã được setup từ đầu file
        
        try:
            logger.info(f"Starting training for {self.model_name} model with Optuna")
            
            # Get prepared data (with caching - shared across all models)
            X_train, X_test, y_train, y_test, self.label_encoder, self.encoders = get_prepared_data(use_cache=True)
            
            # Detect num_classes dynamically
            num_classes = len(self.label_encoder.classes_) if self.label_encoder else 13
            logger.info(f"Detected {num_classes} classes for XGBoost")
            
            self._num_classes = num_classes
            
            original_get_classifier = self.get_classifier
            self.get_classifier = lambda params=None: original_get_classifier(params, num_classes=num_classes)
            
            if isinstance(X_train, pd.DataFrame):
                n_features = X_train.shape[1]
                self._monotonic_constraints = [0] * n_features
                logger.info(f"Monotonic constraints initialized (all zeros - no constraints)")
            else:
                n_features = X_train.shape[1] if hasattr(X_train, 'shape') else 0
                self._monotonic_constraints = [0] * n_features
            
            original_objective = self._objective
            
            def _objective_with_num_classes(trial, X_train, y_train):
                """
                Objective function for XGBoost with num_classes support.
                
                Similar to base _objective but handles dynamic num_classes detection
                and updates classifier in pipeline accordingly.
                """
                try:
                    logger.debug(f"Trial {trial.number}: Starting XGBoost objective")
                    params = self.suggest_hyperparameters(trial)
                    logger.debug(f"Trial {trial.number}: Hyperparameters suggested")
                    pipeline = self.create_pipeline(params)
                    logger.debug(f"Trial {trial.number}: Pipeline created")
                    
                    if hasattr(pipeline, 'named_steps') and 'classifier' in pipeline.named_steps:
                        current_classifier = pipeline.named_steps['classifier']
                        if hasattr(current_classifier, 'get_params'):
                            classifier_params = current_classifier.get_params()
                            classifier_params['num_class'] = num_classes
                            pipeline.named_steps['classifier'] = XGBClassifier(**classifier_params)
                        else:
                            pipeline.named_steps['classifier'] = self.get_classifier(num_classes=num_classes)
                    
                    from sklearn.model_selection import StratifiedKFold
                    from sklearn.metrics import precision_score
                    
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
                            
                            if X_train_fold.empty if isinstance(X_train_fold, pd.DataFrame) else len(X_train_fold) == 0:
                                logger.warning(f"Trial {trial.number}: Empty fold {fold_idx}, skipping")
                                continue
                            
                            if len(y_train_fold) == 0:
                                logger.warning(f"Trial {trial.number}: Empty target fold {fold_idx}, skipping")
                                continue
                            
                            pipeline.fit(X_train_fold, y_train_fold)
                            val_pred = pipeline.predict(X_val_fold)
                            
                            if val_pred is None or len(val_pred) == 0:
                                logger.warning(f"Trial {trial.number}: Empty predictions for fold {fold_idx}")
                                continue
                            
                            score = precision_score(y_val_fold, val_pred, average='macro', zero_division=0)
                            
                            if not np.isfinite(score):
                                logger.warning(f"Trial {trial.number}: Invalid score {score} for fold {fold_idx}, using 0.0")
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
                            logger.warning(f"Trial {trial.number}: Error in fold {fold_idx}: {str(e)}")
                            continue
                    
                    if not scores or len(scores) == 0:
                        logger.warning(f"Trial {trial.number}: No valid scores computed - all folds failed, returning 0.0")
                        return 0.0
                    
                    final_score = np.mean(scores)
                    
                    if not np.isfinite(final_score):
                        logger.warning(f"Trial {trial.number}: Invalid final score {final_score}, returning 0.0")
                        return 0.0
                    
                    return final_score
                
                except optuna.TrialPruned:
                    raise
                except MemoryError as e:
                    logger.error(f"Trial {trial.number}: MemoryError: {str(e)}")
                    raise RuntimeError(f"MemoryError: {str(e)}")
                except Exception as e:
                    logger.warning(f"Trial {trial.number}: Error in objective: {str(e)}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    return 0.0
            
            self._objective = _objective_with_num_classes
            
            # Force sequential Optuna trials for XGBoost to avoid deadlock
            # XGBoost n_jobs=1 (set in hyperparameters) + Optuna sequential = stable
            try:
                from .. import config as config_module_ref
                from ..config import OPTUNA_N_JOBS
            except (ImportError, ValueError):
                import config as config_module_ref
                from config import OPTUNA_N_JOBS
            
            original_optuna_n_jobs = OPTUNA_N_JOBS
            
            # Temporarily override OPTUNA_N_JOBS for XGBoost
            config_module_ref.OPTUNA_N_JOBS = 1
            logger.info("XGBoost: Using sequential Optuna trials (n_jobs=1) and XGBoost n_jobs=1 for stability")
            
            try:
                result = super().train(save_model_flag=save_model_flag, save_plots=save_plots, 
                                   n_trials=n_trials, timeout=timeout)
            finally:
                # Restore original OPTUNA_N_JOBS
                config_module_ref.OPTUNA_N_JOBS = original_optuna_n_jobs
            
            self._objective = original_objective
            
            if hasattr(self.best_model, '_use_monotonic_constraints') and self.best_model._use_monotonic_constraints:
                if hasattr(self.best_model, 'named_steps') and 'classifier' in self.best_model.named_steps:
                    classifier = self.best_model.named_steps['classifier']
                    if hasattr(classifier, 'get_params'):
                        classifier_params = classifier.get_params()
                        classifier_params['monotonic_constraints'] = self._monotonic_constraints
                        new_classifier = XGBClassifier(**classifier_params)
                        self.best_model.named_steps['classifier'] = new_classifier
                        self.best_model.fit(X_train, y_train)
                        logger.info("Applied monotonic constraints to XGBoost model and refitted")
            
            return result
        
        except Exception as e:
            logger.error(f"Error during training {self.model_name}: {str(e)}")
            raise


if __name__ == "__main__":
    from pathlib import Path
    from datetime import datetime
    
    project_root = Path(__file__).parent.parent.parent
    log_file = project_root / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    setup_logging(
        log_file=log_file,
        log_to_console=True,
        log_to_file=True,
        force_reconfigure=True
    )
    logger.info(f"Starting XGBoost training - Log file: {log_file}")
    
    trainer = XGBoostTrainer()
    results = trainer.train()
    logger.info(f"Training completed! Best CV score: {results['best_score']:.4f}")


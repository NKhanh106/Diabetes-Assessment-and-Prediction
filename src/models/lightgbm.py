"""
LightGBM Model Training Script
LightGBM Model Training Script
"""
import optuna
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from lightgbm import LGBMClassifier
from .base_trainer import BaseModelTrainer

# Import centralized logging
try:
    from ..logging_config import get_logger, setup_logging
except ImportError:
    from logging_config import get_logger, setup_logging

logger = get_logger(__name__)


class LightGBMTrainer(BaseModelTrainer):
    """Trainer for LightGBM model"""
    
    def __init__(self):
        super().__init__(model_name='lightgbm', use_scaler=False)  # Tree-based models don't need scaling
    
    def get_classifier(self, params: dict = None, num_classes: Optional[int] = None):
        """
        Return LightGBM classifier
        LightGBM automatically detects num_class from data, but we can set it explicitly
        """
        # LightGBM automatically detects num_class, but we can set it for consistency
        num_class = num_classes if num_classes else None
        
        return LGBMClassifier(
            objective='multiclass',
            num_class=num_class,  # None = auto-detect
            verbose=-1,  # Suppress output
            force_col_wise=True  # For better compatibility
        )
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict:
        """
        Hyperparameter search space for LightGBM
        
        Returns:
            Dictionary of hyperparameters with 'classifier__' prefix
        """
        n_estimators = trial.suggest_int('n_estimators', 200, 1000, step=50)
        
        num_leaves = trial.suggest_int('num_leaves', 31, 127, step=8)
        
        max_depth = trial.suggest_int('max_depth', 4, 12, step=1)
        
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        
        feature_fraction = trial.suggest_float('feature_fraction', 0.7, 1.0, step=0.05)
        
        bagging_fraction = trial.suggest_float('bagging_fraction', 0.7, 1.0, step=0.05)
        bagging_freq = trial.suggest_int('bagging_freq', 1, 5, step=1)
        
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 10, 30, step=5)
        min_gain_to_split = trial.suggest_float('min_gain_to_split', 0.0, 5.0, step=0.5)
        
        lambda_l1 = trial.suggest_float('lambda_l1', 0, 10, step=0.5)
        lambda_l2 = trial.suggest_float('lambda_l2', 0, 10, step=0.5)
        
        boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'gbdt', 'gbdt', 'dart', 'goss'])
        
        try:
            from ..config import N_JOBS
        except (ImportError, ValueError):
            from config import N_JOBS

        params = {
            'classifier__n_estimators': n_estimators,
            'classifier__num_leaves': num_leaves,
            'classifier__max_depth': max_depth,
            'classifier__learning_rate': learning_rate,
            'classifier__feature_fraction': feature_fraction,
            'classifier__bagging_fraction': bagging_fraction,
            'classifier__bagging_freq': bagging_freq,
            'classifier__min_data_in_leaf': min_data_in_leaf,
            'classifier__min_gain_to_split': min_gain_to_split,
            'classifier__lambda_l1': lambda_l1,
            'classifier__lambda_l2': lambda_l2,
            'classifier__boosting_type': boosting_type,
            'classifier__random_state': 42,
            'classifier__n_jobs': N_JOBS,
            'classifier__verbose': -1
        }
        
        # DART-specific parameters (Dropouts meet Multiple Additive Regression Trees)
        if boosting_type == 'dart':
            drop_rate = trial.suggest_float('drop_rate', 0.1, 0.3, step=0.05)
            max_drop = trial.suggest_int('max_drop', 20, 40, step=5)
            skip_drop = trial.suggest_float('skip_drop', 0.4, 0.6, step=0.1)
            uniform_drop = trial.suggest_categorical('uniform_drop', [True, True, False])
            
            params['classifier__drop_rate'] = drop_rate
            params['classifier__max_drop'] = max_drop
            params['classifier__skip_drop'] = skip_drop
            params['classifier__uniform_drop'] = uniform_drop
        
        # GOSS-specific parameters (Gradient-based One-Side Sampling)
        elif boosting_type == 'goss':
            top_rate = trial.suggest_float('top_rate', 0.1, 0.3, step=0.05)
            other_rate = trial.suggest_float('other_rate', 0.1, 0.3, step=0.05)
            
            params['classifier__top_rate'] = top_rate
            params['classifier__other_rate'] = other_rate
        
        # Class weight for imbalanced data
        params['classifier__class_weight'] = trial.suggest_categorical('class_weight', [None, 'balanced'])
        
        return params
    
    def create_pipeline(self, classifier_params: Optional[Dict[str, Any]] = None):
        """
        Create pipeline with LightGBM classifier
        LightGBM automatically detects num_class from training data
        """
        from sklearn.pipeline import Pipeline
        
        # Copy params to avoid side effects
        if classifier_params:
            params_copy = classifier_params.copy()
        else:
            params_copy = {}
        
        steps = []
        
        # LightGBM doesn't need scaler (tree-based)
        classifier = self.get_classifier()
        steps.append(('classifier', classifier))
        
        pipeline = Pipeline(steps)
        
        # Set classifier parameters
        if params_copy:
            classifier_only_params = {k: v for k, v in params_copy.items() 
                                    if k.startswith('classifier__')}
            
            # Filter out invalid parameters based on boosting_type
            valid_params = {}
            boosting_type = params_copy.get('classifier__boosting_type', 'gbdt')
            
            # DART-specific params
            dart_params = ['classifier__drop_rate', 'classifier__max_drop', 
                          'classifier__skip_drop', 'classifier__uniform_drop']
            # GOSS-specific params
            goss_params = ['classifier__top_rate', 'classifier__other_rate']
            
            for key, value in classifier_only_params.items():
                # DART params only valid with dart boosting_type
                if key in dart_params and boosting_type != 'dart':
                    continue
                # GOSS params only valid with goss boosting_type
                if key in goss_params and boosting_type != 'goss':
                    continue
                
                # Bagging params INVALID with GOSS
                if boosting_type == 'goss' and key in ['classifier__bagging_fraction', 'classifier__bagging_freq']:
                    continue
                
                valid_params[key] = value
            
            if valid_params:
                try:
                    pipeline.set_params(**valid_params)
                except (ValueError, TypeError) as e:
                    # Silently handle parameter errors - don't log warnings
                    # Fallback: remove potentially problematic params
                    safe_params = {k: v for k, v in valid_params.items() 
                                 if k not in dart_params + goss_params}
                    if safe_params:
                        try:
                            pipeline.set_params(**safe_params)
                        except Exception:
                            pass  # Use defaults silently
        
        return pipeline
    
    def train(self, save_model_flag: bool = True, save_plots: bool = True, 
              n_trials: Optional[int] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Override train method to handle dynamic num_class detection for LightGBM
        Similar to XGBoost but LightGBM handles it more automatically
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import precision_score
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner
        from optuna.trial import TrialState
        import warnings
        from datetime import datetime
        import uuid
        import logging
        
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
            from config import (
                RANDOM_STATE, CV_FOLDS, SCORING_METRIC,
                N_TRIALS, OPTUNA_TIMEOUT, MODEL_TRIALS
            )
            import config as config_module
            from utils import (
                get_prepared_data, evaluate_model, save_model
            )
        
        # Setup logging if needed
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
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {self.model_name.upper()} model")
        logger.info(f"{'='*60}")
        
        # Load preprocessed training data
        X_train, X_test, y_train, y_test, self.label_encoder, self.encoders = get_prepared_data(use_cache=True)
        
        # Detect num_classes from training data
        num_classes = len(np.unique(y_train))
        logger.info(f"Detected {num_classes} classes in training data")
        
        if n_trials is None:
            n_trials = MODEL_TRIALS.get(self.model_name, N_TRIALS)
        timeout = timeout or OPTUNA_TIMEOUT
        
        optuna_logger = logging.getLogger('optuna')
        optuna_logger.setLevel(logging.WARNING)
        
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
        
        logger.info(f"Created new Optuna study: {unique_study_name}")
        logger.info(f"Starting Optuna optimization: {n_trials} trials for {self.model_name}")
        
        # Create objective function with num_classes
        def _objective_with_num_classes(trial: optuna.Trial) -> float:
            """
            Objective function with dynamic num_classes detection
            LightGBM automatically detects num_class, but we can set it explicitly
            """
            try:
                params = self.suggest_hyperparameters(trial)
                pipeline = self.create_pipeline(params)
                
                # Set num_class explicitly (optional, LightGBM can auto-detect)
                if hasattr(pipeline, 'named_steps') and 'classifier' in pipeline.named_steps:
                    current_classifier = pipeline.named_steps['classifier']
                    if hasattr(current_classifier, 'set_params'):
                        # LightGBM can auto-detect, but we set it for consistency
                        current_classifier.set_params(num_class=num_classes)
                
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
                            logger.debug(f"Trial {trial.number}: Empty fold {fold_idx}, skipping")
                            continue
                        
                        if len(y_train_fold) == 0:
                            logger.debug(f"Trial {trial.number}: Empty target fold {fold_idx}, skipping")
                            continue
                        
                        # Fit pipeline with error handling
                        try:
                            pipeline.fit(X_train_fold, y_train_fold)
                        except (ValueError, TypeError, AttributeError) as e:
                            logger.debug(f"Trial {trial.number}: Pipeline fit error in fold {fold_idx}: {str(e)}")
                            continue
                        except MemoryError as e:
                            logger.error(f"Trial {trial.number}: MemoryError during fit in fold {fold_idx}: {str(e)}")
                            raise RuntimeError(f"MemoryError: {str(e)}")
                        
                        # Predict with error handling
                        try:
                            val_pred = pipeline.predict(X_val_fold)
                        except (ValueError, AttributeError) as e:
                            logger.debug(f"Trial {trial.number}: Prediction error in fold {fold_idx}: {str(e)}")
                            continue
                        
                        if val_pred is None or len(val_pred) == 0:
                            logger.debug(f"Trial {trial.number}: Empty predictions for fold {fold_idx}")
                            continue
                        
                        # Calculate score with validation
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
                        logger.debug(f"Trial {trial.number}: Error in fold {fold_idx}: {str(e)}")
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
        
        # Optimize hyperparameters
        study.optimize(
            _objective_with_num_classes,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=config_module.OPTUNA_N_JOBS,
            show_progress_bar=True
        )
        
        optuna_logger.setLevel(logging.INFO)
        
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not completed_trials:
            raise RuntimeError("No completed trials. All trials failed or were pruned.")
        
        best_trial = study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value
        
        logger.info(f"\nBest trial: {best_trial.number}")
        logger.info(f"Best CV score: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Train final model with best parameters
        logger.info("\nTraining final model with best parameters...")
        best_params_full = self.suggest_hyperparameters(study.best_trial)
        base_pipeline = self.create_pipeline(best_params_full)
        
        # Set num_class for final model
        if hasattr(base_pipeline, 'named_steps') and 'classifier' in base_pipeline.named_steps:
            classifier = base_pipeline.named_steps['classifier']
            classifier.set_params(num_class=num_classes)
        
        self.best_model = base_pipeline
        self.best_model.fit(X_train, y_train)
        
        self.best_model.best_params_ = best_params
        self.best_model.best_score_ = best_score
        
        # Evaluate on test set
        logger.info("\nEvaluating on test set...")
        test_results = evaluate_model(self.best_model, X_test, y_test, self.label_encoder)
        
        # Save model
        if save_model_flag:
            save_model(self.best_model, self.model_name, self.label_encoder, self.encoders)
        
        # Save plots
        if save_plots:
            try:
                from ..utils import save_evaluation_plots
                save_evaluation_plots(test_results, self.model_name)
            except Exception as e:
                logger.debug(f"Could not save plots: {str(e)}")
        
        results = {
            'best_params': best_params,
            'best_score': best_score,
            'test_results': test_results,
            'study': study
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training completed for {self.model_name.upper()}")
        logger.info(f"Best CV score: {best_score:.4f}")
        logger.info(f"{'='*60}")
        
        return results


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
    logger.info(f"Starting LightGBM training - Log file: {log_file}")
    
    trainer = LightGBMTrainer()
    results = trainer.train()
    logger.info(f"Training completed! Best CV score: {results['best_score']:.4f}")


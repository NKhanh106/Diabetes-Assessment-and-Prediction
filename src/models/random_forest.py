"""
Random Forest Model Training Script
Random Forest Model Training Script
"""
import optuna
import numpy as np
from typing import Optional, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from .base_trainer import BaseModelTrainer

# Import centralized logging
try:
    from ..logging_config import get_logger, setup_logging
except ImportError:
    from logging_config import get_logger, setup_logging

logger = get_logger(__name__)


class RandomForestTrainer(BaseModelTrainer):
    """Trainer for Random Forest model"""
    
    def __init__(self):
        super().__init__(model_name='randomforest', use_scaler=False)  # Tree-based models don't need scaling
    
    def get_classifier(self, params: dict = None):
        """Return Random Forest classifier"""
        return RandomForestClassifier()
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict:
        """
        Hyperparameter search space for Random Forest
        
        Returns:
            Dictionary of hyperparameters with 'classifier__' prefix
        """
        n_estimators = trial.suggest_int('n_estimators', 200, 800, step=50)
        
        max_depth_choice = trial.suggest_categorical('max_depth_type', ['limited', 'limited', 'limited', 'unlimited'])
        if max_depth_choice == 'limited':
            max_depth = trial.suggest_int('max_depth', 10, 30, step=2)
        else:
            max_depth = None
        
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'sqrt', 'log2', 'log2', None, 0.6, 0.7, 0.8])
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10, step=1)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
        criterion = trial.suggest_categorical('criterion', ['gini', 'gini', 'gini', 'entropy', 'log_loss'])
        bootstrap = trial.suggest_categorical('bootstrap', [True, True, True, False])
        max_samples_value = trial.suggest_float('max_samples', 0.7, 1.0, step=0.05)
        max_samples = max_samples_value if bootstrap else None
        
        try:
            from ..config import N_JOBS
        except (ImportError, ValueError):
            from config import N_JOBS
        
        params = {
            'classifier__n_estimators': n_estimators,
            'classifier__max_depth': max_depth,
            'classifier__max_features': max_features,
            'classifier__min_samples_split': min_samples_split,
            'classifier__min_samples_leaf': min_samples_leaf,
            'classifier__criterion': criterion,
            'classifier__bootstrap': bootstrap,
            'classifier__oob_score': False,
            'classifier__class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            'classifier__n_jobs': N_JOBS,
            'classifier__random_state': 42
        }
        
        if max_samples is not None:
            params['classifier__max_samples'] = max_samples
        
        return params


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
    logger.info(f"Starting Random Forest training - Log file: {log_file}")
    
    trainer = RandomForestTrainer()
    results = trainer.train()
    logger.info(f"Training completed! Best CV score: {results['best_score']:.4f}")


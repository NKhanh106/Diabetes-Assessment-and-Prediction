"""
Logistic Regression Model Training Script
"""
import optuna
import numpy as np
from typing import Optional, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from .base_trainer import BaseModelTrainer

# Import centralized logging
try:
    from ..logging_config import get_logger, setup_logging
except ImportError:
    from logging_config import get_logger, setup_logging

logger = get_logger(__name__)


class LogisticRegressionTrainer(BaseModelTrainer):
    """Trainer for Logistic Regression model"""
    
    def __init__(self):
        super().__init__(model_name='logistic', use_scaler=True)
    
    def get_classifier(self, params: dict = None):
        """Return Logistic Regression classifier"""
        return LogisticRegression()
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict:
        """
        Suggest hyperparameters for Logistic Regression.
        
        Returns:
            Dictionary of hyperparameters with 'classifier__' prefix for pipeline
        """
        use_polynomial_features = trial.suggest_categorical('use_polynomial_features', [True, True, True, False])
        polynomial_degree = 2
        interaction_only = trial.suggest_categorical('interaction_only', [True, True, False])
        include_bias = trial.suggest_categorical('include_bias', [True, False])
        
        penalty = trial.suggest_categorical('penalty', ['l2', 'l2', 'l2', 'elasticnet', 'l1', 'none'])
        penalty_value = None if penalty == 'none' else penalty
        
        C = trial.suggest_float('C', 0.1, 50, log=True) if penalty != 'none' else None
        
        params = {
            'use_polynomial_features': use_polynomial_features,
            'polynomial_degree': polynomial_degree,
            'interaction_only': interaction_only,
            'include_bias': include_bias,
            'classifier__penalty': penalty_value,
            'classifier__class_weight': trial.suggest_categorical('class_weight', [None, 'balanced', 'balanced']),
            'classifier__max_iter': trial.suggest_int('max_iter', 2000, 4000, step=500),
            'classifier__tol': trial.suggest_float('tol', 1e-4, 1e-3, log=True)
        }
        
        if C is not None:
            params['classifier__C'] = C
        
        all_solvers = ['liblinear', 'lbfgs', 'newton-cg', 'saga']
        suggested_solver = trial.suggest_categorical('solver', all_solvers)
        
        solver_map = {
            'l1': ['liblinear', 'saga'],
            'l2': ['lbfgs', 'newton-cg', 'saga'],
            'elasticnet': ['saga'],
            'none': ['lbfgs', 'newton-cg', 'saga'],
            None: ['lbfgs', 'newton-cg', 'saga']
        }
        
        compatible_solvers = solver_map.get(penalty, ['lbfgs'])
        if suggested_solver not in compatible_solvers:
            fallback_solver = compatible_solvers[0]
            logger.debug(f"Solver {suggested_solver} not compatible with penalty {penalty}, using {fallback_solver}")
            suggested_solver = fallback_solver
        
        if suggested_solver != 'liblinear':
            try:
                from ..config import N_JOBS
            except (ImportError, ValueError):
                from config import N_JOBS
            # Use 50% of N_JOBS for Logistic Regression to be safe with memory
            params['classifier__n_jobs'] = max(1, N_JOBS // 2)
        
        if penalty == 'elasticnet':
            params['classifier__l1_ratio'] = trial.suggest_float('l1_ratio', 0.2, 0.8, step=0.1)
        
        params['classifier__solver'] = suggested_solver
        
        return params
    
    def create_pipeline(self, classifier_params: Optional[Dict[str, Any]] = None):
        """
        Create sklearn Pipeline for Logistic Regression with optional polynomial features.
        
        Pipeline structure:
        1. StandardScaler (if use_scaler=True)
        2. PolynomialFeatures (optional, degree 2, generates ~1,770 features)
        3. SelectKBest (if polynomial enabled, selects top 250 features)
        4. LogisticRegression classifier
        
        Args:
            classifier_params: Dictionary of hyperparameters with 'classifier__' prefix
            
        Returns:
            Configured Pipeline instance
        """
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import SelectKBest, f_classif
        
        if classifier_params:
            params_copy = classifier_params.copy()
        else:
            params_copy = {}
        
        steps = []
        
        if self.use_scaler:
            steps.append(('scaler', StandardScaler()))
        
        use_polynomial_features = params_copy.get('use_polynomial_features', False)
        polynomial_degree = params_copy.get('polynomial_degree', None)
        interaction_only = params_copy.get('interaction_only', None)
        include_bias = params_copy.get('include_bias', None)
        
        if use_polynomial_features and polynomial_degree is not None:
            if polynomial_degree > 2:
                logger.warning(f"Polynomial degree {polynomial_degree} capped at 2")
                polynomial_degree = 2
            
            steps.append(('polynomial_features', PolynomialFeatures(
                degree=polynomial_degree,
                interaction_only=interaction_only if interaction_only is not None else False,
                include_bias=include_bias if include_bias is not None else True,
                sparse=False
            )))
            
            if polynomial_degree == 2:
                max_features_after_poly = 250
            else:
                max_features_after_poly = 200
                logger.warning(f"Unexpected polynomial degree {polynomial_degree}, using fallback")
            
            steps.append(('feature_selection_after_poly', SelectKBest(
                score_func=f_classif,
                k=max_features_after_poly
            )))
            
            logger.debug(
                f"Polynomial features enabled: degree={polynomial_degree}, "
                f"interaction_only={interaction_only}, max_features={max_features_after_poly}"
            )
        
        steps.append(('classifier', self.get_classifier()))
        
        pipeline = Pipeline(steps)
        
        if params_copy:
            classifier_only_params = {
                k: v for k, v in params_copy.items() 
                if k.startswith('classifier__')
            }
            
            for key, value in classifier_only_params.items():
                if value is not None:
                    if isinstance(value, (int, float)) and not np.isfinite(value):
                        logger.warning(f"Invalid parameter value {key}={value}, skipping")
                        classifier_only_params.pop(key, None)
            
            if classifier_only_params:
                # Filter out l1_ratio if penalty is not elasticnet
                valid_params = {}
                penalty_value = params_copy.get('classifier__penalty', None)
                for key, value in classifier_only_params.items():
                    # l1_ratio only valid with elasticnet penalty
                    if 'l1_ratio' in key and penalty_value != 'elasticnet':
                        continue
                    valid_params[key] = value
                
                if valid_params:
                    try:
                        pipeline.set_params(**valid_params)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error setting classifier parameters: {str(e)}, trying without problematic params")
                        # Fallback: remove potentially problematic params
                        safe_params = {k: v for k, v in valid_params.items() 
                                     if k not in ['classifier__l1_ratio', 'classifier__penalty']}
                        if safe_params:
                            try:
                                pipeline.set_params(**safe_params)
                            except Exception:
                                logger.warning(f"Failed to set even safe parameters, using defaults")
        
        return pipeline


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
    logger.info(f"Starting Logistic Regression training - Log file: {log_file}")
    
    trainer = LogisticRegressionTrainer()
    results = trainer.train()
    logger.info(f"Training completed! Best CV score: {results['best_score']:.4f}")


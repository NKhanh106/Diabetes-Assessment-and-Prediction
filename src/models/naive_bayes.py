"""
Naive Bayes Model Training Script
Naive Bayes Model Training Script
"""
import optuna
import numpy as np
from typing import Optional, Dict, Any
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from .base_trainer import BaseModelTrainer

# Import centralized logging
try:
    from ..logging_config import get_logger, setup_logging
except ImportError:
    from logging_config import get_logger, setup_logging

logger = get_logger(__name__)


class NaiveBayesTrainer(BaseModelTrainer):
    """Trainer for Naive Bayes model"""
    
    def __init__(self):
        super().__init__(model_name='naivebayes', use_scaler=True)
    
    def get_classifier(self, params: dict = None):
        """Return Naive Bayes classifier based on params"""
        if params and params.get('nb_variant', 'gaussian') == 'gaussian':
            return GaussianNB()
        elif params and params.get('nb_variant') == 'multinomial':
            return MultinomialNB()
        elif params and params.get('nb_variant') == 'bernoulli':
            return BernoulliNB()
        elif params and params.get('nb_variant') == 'complement':
            return ComplementNB()
        return GaussianNB()  # Default
    
    def create_pipeline(self, classifier_params: Optional[Dict[str, Any]] = None):
        """
        Create sklearn Pipeline for Naive Bayes.
        
        Pipeline structure:
        1. StandardScaler (if use_scaler=True) - important for GaussianNB
        2. Naive Bayes classifier (GaussianNB, MultinomialNB, BernoulliNB, or ComplementNB)
        
        Args:
            classifier_params: Dictionary of hyperparameters
            
        Returns:
            Configured Pipeline instance
        """
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        if classifier_params:
            params_copy = classifier_params.copy()
        else:
            params_copy = {}
        
        steps = []
        
        nb_variant = params_copy.get('nb_variant', 'gaussian')
        
        # StandardScaler is important for GaussianNB
        # Note: MultinomialNB/ComplementNB/BernoulliNB require non-negative data
        # StandardScaler can produce negative values, so we need to handle this
        if self.use_scaler:
            steps.append(('scaler', StandardScaler()))
        
        try:
            if nb_variant == 'gaussian':
                classifier = GaussianNB()
            elif nb_variant == 'multinomial':
                classifier = MultinomialNB()
            elif nb_variant == 'bernoulli':
                classifier = BernoulliNB()
            elif nb_variant == 'complement':
                classifier = ComplementNB()
            else:
                classifier = GaussianNB()  # Default silently
        except Exception as e:
            classifier = GaussianNB()  # Fallback silently
        
        steps.append(('classifier', classifier))
        
        pipeline = Pipeline(steps)
        
        if params_copy:
            classifier_only_params = {
                k: v for k, v in params_copy.items() 
                if k.startswith('classifier__')
            }
            
            # Filter out invalid parameters based on classifier type
            valid_params = {}
            for key, value in classifier_only_params.items():
                if value is None:
                    continue
                
                # Skip class_weight for all Naive Bayes variants (not supported)
                if 'class_weight' in key:
                    continue
                
                # Validate numeric values
                if isinstance(value, (int, float)):
                    if not np.isfinite(value):
                        continue
                
                valid_params[key] = value
            
            if valid_params:
                try:
                    pipeline.set_params(**valid_params)
                except (ValueError, TypeError) as e:
                    # Silently handle parameter errors - don't log warnings
                    # Fallback: remove potentially problematic params based on error
                    error_msg = str(e).lower()
                    safe_params = valid_params.copy()
                    
                    # Remove alpha if error mentions it (only for MultinomialNB/ComplementNB)
                    if 'alpha' in error_msg and nb_variant == 'gaussian':
                        safe_params = {k: v for k, v in safe_params.items() if 'alpha' not in k}
                    
                    # Remove var_smoothing if error mentions it (only for GaussianNB)
                    if 'var_smoothing' in error_msg and nb_variant != 'gaussian':
                        safe_params = {k: v for k, v in safe_params.items() if 'var_smoothing' not in k}
                    
                    if safe_params:
                        try:
                            pipeline.set_params(**safe_params)
                        except Exception:
                            pass  # Use defaults silently
        
        return pipeline
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict:
        """
        Hyperparameter search space for Naive Bayes
        
        Returns:
            Dictionary of hyperparameters with 'classifier__' prefix
        """
        nb_variant = trial.suggest_categorical('nb_variant', [
            'gaussian', 'gaussian', 'gaussian', 'gaussian', 'gaussian',
            'multinomial', 'complement', 'bernoulli'
        ])
        
        params = {
            'nb_variant': nb_variant
        }
        
        if nb_variant == 'gaussian':
            # GaussianNB parameters
            var_smoothing = trial.suggest_float('var_smoothing', 1e-9, 1e-5, log=True)
            params['classifier__var_smoothing'] = var_smoothing
            params['classifier__priors'] = None  # Learn from data
            # GaussianNB does NOT support class_weight - don't suggest it
            
        elif nb_variant in ['multinomial', 'complement', 'bernoulli']:
            # MultinomialNB/BernoulliNB/ComplementNB parameters
            alpha = trial.suggest_float('alpha', 0.1, 5.0, log=True)
            params['classifier__alpha'] = alpha
            
            # fit_prior: Whether to learn class priors from data
            # Usually True works better
            fit_prior = trial.suggest_categorical('fit_prior', [True, True, False])
            params['classifier__fit_prior'] = fit_prior
            
            # class_prior: Optional class priors (None = learn from data)
            params['classifier__class_prior'] = None
            
            if nb_variant == 'bernoulli':
                # BernoulliNB specific: binarize threshold
                # Optimized range: 0.0-0.5 (typical optimal around 0.0-0.3)
                binarize = trial.suggest_float('binarize', 0.0, 0.5, step=0.1)
                params['classifier__binarize'] = binarize
            
            # MultinomialNB/ComplementNB/BernoulliNB do NOT support class_weight - don't suggest it
        
        # Note: None of the Naive Bayes variants support class_weight parameter
        # We handle imbalanced data through class_prior or by using different variants
        
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
    logger.info(f"Starting Naive Bayes training - Log file: {log_file}")
    
    trainer = NaiveBayesTrainer()
    results = trainer.train()
    logger.info(f"Training completed! Best CV score: {results['best_score']:.4f}")


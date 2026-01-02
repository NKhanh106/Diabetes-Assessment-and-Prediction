"""
K-Nearest Neighbors (KNN) Model Training Script
K-Nearest Neighbors (KNN) Model Training Script
"""
import optuna
import numpy as np
from typing import Optional, Dict, Any
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from .base_trainer import BaseModelTrainer

# Import centralized logging
try:
    from ..logging_config import get_logger, setup_logging
except ImportError:
    from logging_config import get_logger, setup_logging

logger = get_logger(__name__)


class KNNTrainer(BaseModelTrainer):
    """Trainer for K-Nearest Neighbors model with feature selection"""
    
    def __init__(self):
        super().__init__(model_name='knn', use_scaler=True)
        self.feature_selector = None
        self.use_feature_selection = True
    
    def get_classifier(self, params: dict = None):
        """Return KNN classifier"""
        return KNeighborsClassifier()
    
    def create_pipeline(self, classifier_params: Optional[dict] = None):
        """
        Create sklearn Pipeline for KNN with optional feature selection.
        
        Pipeline structure:
        1. StandardScaler (if use_scaler=True)
        2. Feature selection: SelectKBest or PCA (optional)
        3. KNeighborsClassifier
        
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
        
        if self.use_scaler:
            steps.append(('scaler', StandardScaler()))
        
        use_feature_selection = params_copy.get('use_feature_selection', False)
        feature_selection_method = params_copy.get('feature_selection_method', None)
        n_features = params_copy.get('n_features', None)
        
        if use_feature_selection and feature_selection_method and n_features is not None:
            try:
                if feature_selection_method == 'selectkbest':
                    n_features_int = int(n_features)
                    if n_features_int < 1:
                        logger.warning(f"Invalid n_features {n_features_int}, using minimum 1")
                        n_features_int = 1
                    steps.append(('feature_selection', SelectKBest(
                        score_func=f_classif,
                        k=n_features_int
                    )))
                elif feature_selection_method == 'pca':
                    pca_variance = float(n_features)
                    if not np.isfinite(pca_variance) or pca_variance < 0.1:
                        logger.warning(f"Invalid PCA variance {pca_variance}, using 0.85")
                        pca_variance = 0.85
                    elif pca_variance > 1.0:
                        logger.warning(f"Invalid PCA variance {pca_variance}, using 0.99")
                        pca_variance = 0.99
                    steps.append(('feature_selection', PCA(
                        n_components=pca_variance,
                        random_state=42
                    )))
            except (ValueError, TypeError) as e:
                logger.warning(f"Error in feature selection setup: {str(e)}, skipping feature selection")
        
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
                pipeline.set_params(**classifier_only_params)
        
        return pipeline
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict:
        """
        Hyperparameter search space for KNN
        
        Returns:
            Dictionary of hyperparameters with 'classifier__' prefix
        """
        use_feature_selection = trial.suggest_categorical('use_feature_selection', [True, False])
        feature_selection_method = trial.suggest_categorical('feature_selection_method', ['selectkbest', 'pca'])
        n_features = None
        
        if use_feature_selection:
            if feature_selection_method == 'selectkbest':
                n_features = trial.suggest_int('n_features_kbest', 10, 58, step=2)
            else:
                pca_variance = trial.suggest_float('pca_variance', 0.75, 0.99, step=0.01)
                n_features = pca_variance
        else:
            # Still suggest for Optuna tracking even if not used
            _ = trial.suggest_int('n_features_kbest', 10, 58, step=2)
            _ = trial.suggest_float('pca_variance', 0.75, 0.99, step=0.01)
        
        n_neighbors = trial.suggest_int('n_neighbors', 50, 500, step=10)
        
        # Weights
        weights = trial.suggest_categorical('weights', ['distance', 'uniform'])
        
        # Algorithm: auto lets sklearn choose the most efficient
        algorithm = 'auto'
        
        metric = trial.suggest_categorical('metric', [
            'euclidean', 'euclidean',
            'manhattan', 'minkowski', 'chebyshev'
        ])
        
        p_value = None
        if metric == 'minkowski':
            p_value = trial.suggest_int('p', 1, 3)
        
        # Standard leaf size
        leaf_size = trial.suggest_int('leaf_size', 10, 50, step=10)
        
        params = {
            'classifier__n_neighbors': n_neighbors,
            'classifier__weights': weights,
            'classifier__metric': metric,
            'classifier__algorithm': algorithm,
            'classifier__leaf_size': leaf_size,
        }
        
        try:
            from ..config import N_JOBS
        except (ImportError, ValueError):
            from config import N_JOBS
            
        params['classifier__n_jobs'] = N_JOBS
        
        # Add non-classifier params
        params['use_feature_selection'] = use_feature_selection
        params['feature_selection_method'] = feature_selection_method
        params['n_features'] = n_features

        
        if p_value is not None:
            params['classifier__p'] = p_value
        
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
    logger.info(f"Starting KNN training - Log file: {log_file}")
    
    trainer = KNNTrainer()
    results = trainer.train()
    logger.info(f"Training completed! Best CV score: {results['best_score']:.4f}")


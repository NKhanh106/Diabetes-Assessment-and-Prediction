"""
Script to train all models at once with optional parallel training
"""
import sys
from pathlib import Path
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# Add src to path (file is now in src/, so parent is project root)
project_root = Path(__file__).parent.parent
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from models import (
    KNNTrainer, SVMTrainer, RandomForestTrainer,
    LogisticRegressionTrainer, XGBoostTrainer
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def _train_single_model(model_name: str, trainer_class, n_trials: int, timeout: int, save_plots: bool):
    """
    Train a single model (used for parallel execution)
    
    Args:
        model_name: Name of the model
        trainer_class: Trainer class to instantiate
        n_trials: Number of Optuna trials
        timeout: Timeout in seconds
        save_plots: Whether to save plots
        
    Returns:
        Tuple of (model_name, result_dict)
    """
    try:
        trainer = trainer_class()
        start_time = datetime.now()
        
        result = trainer.train(
            save_model_flag=True,
            save_plots=save_plots,
            n_trials=n_trials,
            timeout=timeout
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"✓ {model_name} completed - Score: {result['best_score']:.4f} | Duration: {duration:.1f}s")
        
        return (model_name, {
            'best_params': result['best_params'],
            'best_score': result['best_score'],
            'duration_seconds': duration,
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"\nError training {model_name}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return (model_name, {
            'status': 'failed',
            'error': str(e)
        })


def train_all_models(n_trials=None, timeout=None, save_plots=True, parallel=False, max_workers=None):
    """
    Train all models sequentially or in parallel
    
    Args:
        n_trials: Number of Optuna trials for each model (None = use config default)
        timeout: Timeout in seconds for each model (None = use config default)
        save_plots: Whether to save evaluation plots
        parallel: Whether to train models in parallel (default: False)
        max_workers: Maximum number of parallel workers (None = auto-detect)
    """
    trainers = [
        ('KNN', KNNTrainer),
        ('SVM', SVMTrainer),
        ('Random Forest', RandomForestTrainer),
        ('Logistic Regression', LogisticRegressionTrainer),
        ('XGBoost', XGBoostTrainer)
    ]
    
    results = {}
    
    logger.info("=" * 60)
    logger.info("Starting training for all models")
    logger.info(f"Number of trials per model: {n_trials or 'default from config'}")
    logger.info(f"Timeout per model: {timeout or 'default from config'}")
    logger.info(f"Parallel training: {parallel}")
    if parallel:
        max_workers = max_workers or min(cpu_count(), len(trainers))
        logger.info(f"Max parallel workers: {max_workers}")
    logger.info("=" * 60)
    
    if parallel:
        # Parallel training: each process has its own memory space, so data cache is not shared
        # Sequential training is more efficient for data caching
        logger.warning("Parallel training enabled. Data cache is not shared across processes in parallel mode.")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_train_single_model, model_name, trainer_class, n_trials, timeout, save_plots): model_name
                for model_name, trainer_class in trainers
            }
            
            for future in as_completed(futures):
                model_name, result = future.result()
                results[model_name] = result
                if result['status'] != 'success':
                    logger.error(f"✗ {model_name} failed: {result.get('error', 'Unknown error')}")
    else:
        # Sequential training: allows efficient data cache sharing across all models
        for model_name, trainer_class in trainers:
            _, result = _train_single_model(model_name, trainer_class, n_trials, timeout, save_plots)
            results[model_name] = result
            
            if result['status'] != 'success':
                logger.error(f"✗ {model_name} failed: {result.get('error', 'Unknown error')}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    
    for model_name, result in results.items():
        if result['status'] == 'success':
            logger.info(f"{model_name:25s} | Score: {result['best_score']:.4f} | "
                       f"Duration: {result['duration_seconds']:.1f}s")
        else:
            logger.error(f"{model_name:25s} | FAILED: {result.get('error', 'Unknown error')}")
    
    logger.info("=" * 60)
    
    # Find best model
    successful_models = {k: v for k, v in results.items() if v['status'] == 'success'}
    if successful_models:
        best_model = max(successful_models.items(), key=lambda x: x[1]['best_score'])
        logger.info(f"\nBest model: {best_model[0]} with score: {best_model[1]['best_score']:.4f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train all ML models')
    parser.add_argument('--n-trials', type=int, default=None,
                       help='Number of Optuna trials per model (default: from config)')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Timeout in seconds per model (default: from config)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip saving evaluation plots')
    parser.add_argument('--parallel', action='store_true',
                       help='Train models in parallel (faster but uses more resources)')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of parallel workers (default: auto-detect)')
    
    args = parser.parse_args()
    
    train_all_models(
        n_trials=args.n_trials,
        timeout=args.timeout,
        save_plots=not args.no_plots,
        parallel=args.parallel,
        max_workers=args.max_workers
    )


# Diabetes Assessment and Prediction

A production-ready machine learning system for diabetes type classification using an ensemble of five optimized models with automated hyperparameter tuning.

## Overview

This project implements a comprehensive machine learning pipeline for diabetes type prediction using an ensemble approach. The system employs five distinct algorithms with Optuna-based hyperparameter optimization, advanced preprocessing techniques, and weighted ensemble voting for robust predictions.

### Models

- **K-Nearest Neighbors- **KNN**: 45 (Feature selection and dimensionality reduction enabled)
- **Logistic Regression**: 75 (Polynomial features and interaction terms included)
- **Random Forest**: 90 (High robustness and feature importance analysis)
- **Naive Bayes**: 82 (Efficient for high-dimensional data)
- **XGBoost**: 90 (Gradient boosting with regularization)
- **LightGBM**: 91 (High efficiency and accuracy)boosting with efficient histogram-based algorithms

## Architecture

### Project Structure

```
Diabetes-Assessment-and-Prediction/
├── data/                          # Data storage
│   ├── raw/                       # Raw dataset files
│   └── processed/                 # Preprocessed dataset files
│
├── src/                           # Source code
│   ├── config.py                  # Centralized configuration
│   ├── utils.py                   # Data processing and evaluation utilities
│   ├── train_all_models.py        # Batch training orchestration
│   ├── models/                    # Model implementations
│   │   ├── base_trainer.py        # Abstract base class for all trainers
│   │   ├── knn.py                 # KNN model trainer
│   │   ├── naive_bayes.py         # Naive Bayes trainer
│   │   ├── random_forest.py       # Random Forest trainer
│   │   ├── logistic_regression.py # Logistic Regression trainer
│   │   ├── xgboost.py             # XGBoost trainer
│   │   └── lightgbm.py            # LightGBM trainer
│   ├── preprocessing/             # Data preprocessing modules
│   │   ├── data_processor.py      # Main preprocessing pipeline
│   │   └── advanced_preprocessing.py # Advanced feature engineering
│   └── prediction/                # Prediction module
│       └── predictor.py           # Ensemble prediction system
│
├── models/                        # Trained model artifacts (generated)
│   ├── knn/                       # KNN model files
│   ├── naive_bayes/               # Naive Bayes model files
│   ├── random_forest/             # Random Forest files
│   ├── logistic_regression/       # Logistic Regression files
│   ├── xgboost/                   # XGBoost files
│   └── lightgbm/                  # LightGBM files
│
├── results/                       # Evaluation results (generated)
│   └── *_heatmap.png              # Confusion matrix visualizations
│
├── app/                           # Deployment application
│   └── deploy.py                  # Streamlit web interface
│
└── notebooks/                     # Jupyter notebooks for analysis
```

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Diabetes-Assessment-and-Prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download dataset:**
   
   The dataset is available on Kaggle: [Diabetes Dataset](https://www.kaggle.com/datasets/ankitbatra1210/diabetes-dataset)
   
   **Using Kaggle API:**
   ```bash
   pip install kaggle
   kaggle datasets download -d ankitbatra1210/diabetes-dataset
   unzip diabetes-dataset.zip -d data/raw/
   ```
   
   **Manual download:**
   - Download from Kaggle
   - Place `diabetes_dataset.csv` in `data/raw/` directory

## Usage

### Data Preprocessing

Execute the preprocessing pipeline to prepare raw data:

```bash
python -m src.preprocessing.data_processor
```

This pipeline performs:
- Column type identification (numerical/categorical)
- Categorical feature encoding (OneHotEncoder)
- Advanced preprocessing (outlier handling, feature engineering, multicollinearity removal)
- Data validation and quality checks
- Processed data persistence

### Model Training

**Batch Training (Recommended):**

Train all models with default configuration:
```bash
python src/train_all_models.py
```

Custom training parameters:
```bash
# Specify number of Optuna trials per model
python src/train_all_models.py --n-trials 50

# Set timeout per model (seconds)
python src/train_all_models.py --timeout 3600

# Disable plot generation
python src/train_all_models.py --no-plots

# Enable parallel training
python src/train_all_models.py --parallel --max-workers 4

# Combine options
python src/train_all_models.py --n-trials 50 --timeout 1800 --parallel
```

**Individual Model Training:**

```bash
python -m src.models.knn
python -m src.models.naive_bayes
python -m src.models.random_forest
python -m src.models.logistic_regression
python -m src.models.xgboost
python -m src.models.lightgbm
```

**Programmatic Training:**

```python
from src.models import (
    KNNTrainer, NaiveBayesTrainer, RandomForestTrainer,
    LogisticRegressionTrainer, XGBoostTrainer, LightGBMTrainer
)

trainer = KNNTrainer()
results = trainer.train(
    save_model_flag=True,
    save_plots=True,
    n_trials=50,
    timeout=3600
)
```

### Deployment

Launch the Streamlit web application:

```bash
cd app
streamlit run deploy.py
```

The application provides an interactive interface for diabetes type prediction using the trained ensemble model.

## Technical Features

### Hyperparameter Optimization

- **Optuna Framework**: Tree-structured Parzen Estimator (TPE) sampling
- **Median Pruning**: Early stopping for underperforming trials
- **Model-Specific Configuration**: Optimized trial counts per model type
- **Parallel Execution**: Multi-core optimization support

### Data Preprocessing

- **Automatic Type Detection**: Numerical and categorical column identification
- **Advanced Feature Engineering**: 
  - Outlier detection and handling (IQR, Z-score)
  - Log transformation for skewed features
  - Interaction feature creation
  - Multicollinearity removal
  - Low variance feature elimination
- **Memory Optimization**: Efficient data type conversion and caching

### Model Training

- **Stratified K-Fold Cross-Validation**: Maintains class distribution
- **Intermediate Value Reporting**: Enables Optuna pruning
- **Model Persistence**: Saves trained models with encoders
- **Evaluation Metrics**: Comprehensive classification reports and confusion matrices

### Ensemble Prediction

- **Weighted Voting**: Model-specific weights for ensemble decisions
- **Probability Calibration**: Improved probability estimates for supported models
- **Efficient Loading**: Cached model and encoder loading

## Configuration

Edit `src/config.py` to customize:

- **Training Parameters**: Random state, test split ratio, CV folds
- **Optimization Settings**: Trial counts, timeout, parallel jobs
- **Scoring Metric**: precision_macro, recall_macro, f1_macro
- **Model Weights**: Ensemble voting weights per model
- **Advanced Preprocessing**: Feature engineering options

### Model Weights

Current ensemble configuration:
- KNN: 45
- Logistic Regression: 75
- Random Forest: 90
- Naive Bayes: 82
- XGBoost: 90
- LightGBM: 91

## Dataset

- **Source**: [Kaggle - Diabetes Dataset](https://www.kaggle.com/datasets/ankitbatra1210/diabetes-dataset)
- **Author**: Ankit Batra
- **Raw Data**: `data/raw/diabetes_dataset.csv`
- **Processed Data**: `data/processed/diabetes_dataset_processed.csv` (generated)

The dataset contains features related to diabetes assessment including genetic markers, lifestyle factors, medical history, and clinical test results.

## Performance

The system implements several performance optimizations:

- **Data Caching**: In-memory caching for processed data across models
- **Vectorized Operations**: NumPy-based computations for efficiency
- **Parallel Processing**: Multi-core support for training and optimization
- **Memory Optimization**: Efficient data type conversion and storage

## Evaluation

Model evaluation includes:

- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score (macro/micro/weighted)
- **Confusion Matrices**: Visual heatmaps for each model
- **Cross-Validation Scores**: Stratified K-Fold performance metrics
- **Best Hyperparameters**: Optimal configuration per model

Results are saved in the `results/` directory.

## Medical Disclaimer

⚠️ **Important**: This system is a machine learning prediction tool for research and educational purposes. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for accurate medical assessment and treatment decisions.

## Requirements

See `requirements.txt` for complete dependency list. Key dependencies include:

- scikit-learn
- optuna
- xgboost
- lightgbm
- pandas
- numpy
- streamlit
- matplotlib

## License

[Specify license]

## Contributing

[Specify contribution guidelines]

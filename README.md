# Diabetes Assessment and Prediction

A machine learning project for predicting diabetes types using an ensemble of 5 different models.

## Overview

This project uses an ensemble approach with 5 machine learning models to predict diabetes types:
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Random Forest**
- **Logistic Regression**
- **XGBoost**

The final prediction is made using weighted voting from all 5 models.

## Project Structure

```
Diabetes-Assessment-and-Prediction/
├── data/                          # Data files
│   ├── raw/                       # Raw dataset
│   └── processed/                 # Processed dataset
│
├── src/                           # Source code
│   ├── config.py                  # Configuration
│   ├── utils.py                   # Utility functions
│   ├── models/                    # Model trainers
│   │   ├── base_trainer.py
│   │   ├── knn.py
│   │   ├── svm.py
│   │   ├── random_forest.py
│   │   ├── logistic_regression.py
│   │   └── xgboost.py
│   ├── preprocessing/             # Data preprocessing
│   │   └── data_processor.py
│   └── prediction/                # Prediction module
│       └── predictor.py
│
├── models/                        # Saved trained models (generated)
├── results/                       # Evaluation results (generated)
├── app/                           # Streamlit application
│   └── deploy.py
└── notebooks/                     # Jupyter notebooks (optional)
```

See [STRUCTURE.md](STRUCTURE.md) for detailed structure documentation.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Diabetes-Assessment-and-Prediction
   ```

2. **Download the dataset:**
   
   The dataset is available on Kaggle. You can download it using one of the following methods:
   
   **Option 1: Using Kaggle API (Recommended)**
   ```bash
   # Install Kaggle API (if not already installed)
   pip install kaggle
   
   # Set up Kaggle API credentials (place your kaggle.json in ~/.kaggle/)
   # Download the dataset
   kaggle datasets download -d ankitbatra1210/diabetes-dataset
   
   # Extract and place the CSV file in data/raw/
   unzip diabetes-dataset.zip -d data/raw/
   # Or on Windows:
   # Expand-Archive diabetes-dataset.zip -DestinationPath data/raw/
   ```
   
   **Option 2: Manual Download**
   1. Visit the dataset page: [Diabetes Dataset on Kaggle](https://www.kaggle.com/datasets/ankitbatra1210/diabetes-dataset)
   2. Click "Download" button (requires Kaggle account)
   3. Extract the downloaded file
   4. Place `diabetes_dataset.csv` in the `data/raw/` directory
   
   **Option 3: Direct Download Link**
   - Dataset URL: https://www.kaggle.com/datasets/ankitbatra1210/diabetes-dataset
   - After downloading, ensure the file is named `diabetes_dataset.csv` and placed in `data/raw/`

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 0. Dataset Setup

Make sure you have downloaded the dataset and placed `diabetes_dataset.csv` in the `data/raw/` directory. If you haven't done this yet, refer to the [Installation](#installation) section above.

### 1. Data Preprocessing

First, preprocess the raw data:
```bash
cd src/preprocessing
python data_processor.py
```

This will:
- Load the raw dataset
- Identify numerical and categorical columns
- Encode categorical features using OneHotEncoder
- Save the processed dataset and encoders

### 2. Training Models

**Option 1: Train all models at once (Recommended)**
```bash
cd src
python train_all_models.py
```

Or from project root:
```bash
python src/train_all_models.py
```

With custom options:
```bash
# Train with 100 trials per model
python src/train_all_models.py --n-trials 100

# Train with timeout of 1 hour per model
python src/train_all_models.py --timeout 3600

# Train without saving plots
python src/train_all_models.py --no-plots

# Combine options
python src/train_all_models.py --n-trials 50 --timeout 1800
```

**Option 2: Train individual models**
```bash
cd src/models
python knn.py
python svm.py
python random_forest.py
python logistic_regression.py
python xgboost.py
```

**Option 3: Train programmatically**
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import (
    KNNTrainer, SVMTrainer, RandomForestTrainer,
    LogisticRegressionTrainer, XGBoostTrainer
)

trainers = [
    KNNTrainer(),
    SVMTrainer(),
    RandomForestTrainer(),
    LogisticRegressionTrainer(),
    XGBoostTrainer()
]

for trainer in trainers:
    results = trainer.train()
    print(f"{trainer.model_name}: {results['best_score']:.4f}")
```

### 3. Running the Web Application

Deploy the Streamlit web application:
```bash
cd app
streamlit run deploy.py
```

The application will open in your browser. Fill in the patient information form and click "Submit" to get predictions.

## Features

### Code Quality Improvements

- ✅ **Modular Design**: Separated concerns with config, utils, and base trainer
- ✅ **DRY Principle**: Eliminated code duplication across model files
- ✅ **Error Handling**: Comprehensive error handling and logging
- ✅ **Type Hints**: Added type hints for better code documentation
- ✅ **Logging**: Integrated logging throughout the project
- ✅ **Visualization**: Automatic confusion matrix generation
- ✅ **Efficient Prediction**: Cached models and encoders for faster predictions
- ✅ **Input Validation**: Validates user input in the web application

### Model Training Features

- **Optuna** for intelligent hyperparameter optimization (replaces GridSearchCV)
- Stratified K-Fold cross-validation
- Automatic model evaluation and metrics
- Confusion matrix visualization
- Model persistence with encoders
- Batch training script to train all models at once

### Prediction Features

- Ensemble prediction with weighted voting
- Efficient model loading with caching
- Proper encoder handling for new data
- Error handling and logging

## Configuration

Edit `src/config.py` to customize:
- Random state for reproducibility
- Test/train split ratio
- Cross-validation folds
- Scoring metric (precision_macro, recall_macro, f1_macro)
- Model weights for ensemble voting

## Model Weights

The ensemble uses the following weights:
- KNN: 50
- Logistic Regression: 75
- Random Forest: 90
- SVM: 75
- XGBoost: 90

## Dataset

- **Source**: [Diabetes Dataset on Kaggle](https://www.kaggle.com/datasets/ankitbatra1210/diabetes-dataset)
- **Dataset Name**: Diabetes Dataset by Ankit Batra
- **File Location**: `data/raw/diabetes_dataset.csv`
- **Processed Location**: `data/processed/diabetes_dataset_processed.csv` (generated after preprocessing)

The dataset contains various features related to diabetes assessment including genetic markers, lifestyle factors, medical history, and test results.

## Notes

- ⚠️ **Medical Disclaimer**: This is a machine learning prediction tool. Always consult with healthcare professionals for accurate medical diagnosis.
- The models are trained on a specific dataset and may not generalize to all populations.
- Ensure all required features are provided for accurate predictions.
- Make sure to download the dataset from Kaggle before running preprocessing or training scripts.

## Requirements

See `requirements.txt` for the complete list of dependencies.

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
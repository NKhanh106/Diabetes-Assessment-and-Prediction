# Model Evaluation Results

This directory contains evaluation artifacts generated during model training, including confusion matrix visualizations and performance metrics.

## Contents

### Confusion Matrix Heatmaps

After training, each model generates a dual-panel confusion matrix visualization:

- **Left Panel**: Raw prediction counts
- **Right Panel**: Normalized percentages (row-wise)

**File Naming Convention:**
- `{model_name}_heatmap.png`

**Available Visualizations:**
- `knn_heatmap.png` - K-Nearest Neighbors confusion matrix
- `logisticregression_heatmap.png` - Logistic Regression confusion matrix
- `randomforest_heatmap.png` - Random Forest confusion matrix
- `naivebayes_heatmap.png` - Naive Bayes confusion matrix
- `xgboost_heatmap.png` - XGBoost confusion matrix
- `lightgbm_heatmap.png` - LightGBM confusion matrix

## Confusion Matrix Interpretation

A confusion matrix visualizes classification model performance by comparing predicted vs. actual class labels.

### Matrix Components

- **True Positives (TP)**: Correctly predicted positive cases (diagonal elements)
- **True Negatives (TN)**: Correctly predicted negative cases (diagonal elements)
- **False Positives (FP)**: Incorrectly predicted as positive (off-diagonal, predicted column)
- **False Negatives (FN)**: Incorrectly predicted as negative (off-diagonal, actual row)

### Visualization Features

- **Color Intensity**: Represents prediction frequency
- **Count Display**: Raw prediction counts in left panel
- **Percentage Display**: Normalized percentages in right panel
- **Class Labels**: Actual class names from label encoder

## Generation

Heatmaps are automatically generated during model training when:

1. Training via `train_all_models.py` (default behavior)
2. Individual model training with `save_plots=True`
3. Model evaluation through `evaluate_model()` function

**Command:**
```bash
python src/train_all_models.py
```

## Technical Details

### Generation Process

1. Model predictions on test set
2. Confusion matrix computation using scikit-learn
3. Dual-panel visualization creation
4. High-resolution PNG export (300 DPI)
5. File persistence to `results/` directory

### Visualization Configuration

- **Figure Size**: 20×8 inches (dual panels)
- **Color Schemes**: 
  - Left panel: Blues colormap
  - Right panel: Oranges colormap
- **Format**: PNG with 300 DPI resolution
- **Metadata**: Model name, accuracy, and F1-macro score included in title

## Usage

These visualizations serve multiple purposes:

- **Performance Analysis**: Identify class-specific prediction patterns
- **Model Comparison**: Compare confusion matrices across models
- **Error Analysis**: Locate systematic misclassification patterns
- **Documentation**: Visual representation of model capabilities

## File Management

- Files are automatically overwritten on each training run
- Historical versions can be preserved by renaming before retraining
- Files are excluded from version control (see `.gitignore`)

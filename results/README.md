# Model Evaluation Results

This directory contains visualization results from model evaluation, specifically confusion matrix heatmaps.

## Files

After training each model, a confusion matrix heatmap is automatically saved here with the naming convention:
- `{model_name}_heatmap.png`

For example:
- `knn_heatmap.png`
- `logisticregression_heatmap.png`
- `randomforest_heatmap.png`
- `svm_heatmap.png`
- `xgboost_heatmap.png`

## What is a Confusion Matrix?

A confusion matrix is a table that visualizes the performance of a classification model. It shows:
- **True Positives (TP)**: Correctly predicted positive cases
- **True Negatives (TN)**: Correctly predicted negative cases
- **False Positives (FP)**: Incorrectly predicted as positive
- **False Negatives (FN)**: Incorrectly predicted as negative

The heatmap uses color intensity to represent the count of predictions in each category, making it easy to identify where the model performs well and where it struggles.

## Usage

These images are automatically generated when you run:
```bash
python src/train_all_models.py
```

Or when training individual models with the `save_plots=True` parameter.


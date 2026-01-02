"""
Advanced preprocessing techniques for maximum model performance
Includes outlier detection, feature engineering, scaling options, and more
Refactored to support separate Fit (Train) and Transform (Test) phases to prevent Data Leakage.
"""
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from sklearn.preprocessing import (
    RobustScaler, MinMaxScaler, QuantileTransformer, 
    PowerTransformer, StandardScaler
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, 
    VarianceThreshold, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def detect_outliers_iqr(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                       factor: float = 1.5) -> Dict[str, Dict[str, float]]:
    """
    Calculate outlier bounds using Interquartile Range (IQR) method (Fit phase)
    
    Args:
        df: Input DataFrame
        columns: List of columns to check (None = all numerical)
        factor: IQR factor (default 1.5)
        
    Returns:
        Dictionary mapping column names to bounds {'lower': float, 'upper': float}
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    bounds = {}
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        bounds[col] = {'lower': lower_bound, 'upper': upper_bound}
    
    return bounds


def handle_outliers(df: pd.DataFrame, method: str = 'cap', 
                   columns: Optional[List[str]] = None,
                   bounds: Optional[Dict[str, Dict[str, float]]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handle outliers in numerical columns using pre-calculated bounds or calculating them
    
    Args:
        df: Input DataFrame
        method: 'cap' (cap at bounds), 'remove' (remove rows - TRAIN ONLY), 'winsorize'
        columns: Columns to process (None = all numerical)
        bounds: Pre-computed bounds (from train set). If None, will calculate.
        
    Returns:
        Tuple of (Processed DataFrame, bounds dictionary)
    """
    df_processed = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # If using 'remove', we just drop rows based on current data (usually only for train)
    # BE CAREFUL: 'remove' shouldn't be used on Test set typically, but if requested we do it.
    if method == 'remove':
        if bounds is None: # Calculate indices
             # Note: For 'remove', we typically re-detect roughly or just don't support it for test
             # Here we implement simple IQR detect and remove
             curr_bounds = detect_outliers_iqr(df, columns)
        else:
             curr_bounds = bounds

        indices_to_drop = set()
        for col, bound in curr_bounds.items():
            if col in df.columns:
                 mask = (df[col] < bound['lower']) | (df[col] > bound['upper'])
                 indices_to_drop.update(df[mask].index.tolist())
        
        if indices_to_drop:
            df_processed = df_processed.drop(index=list(indices_to_drop))
        
        return df_processed, curr_bounds

    # For 'cap' or 'winsorize' (winsorize is similar to cap but with percentiles, here mapping to bounds)
    if bounds is None:
        bounds = detect_outliers_iqr(df, columns)
    
    for col, bound in bounds.items():
        if col not in df.columns:
            continue
            
        df_processed[col] = df_processed[col].clip(lower=bound['lower'], upper=bound['upper'])
            
    return df_processed, bounds


def apply_scaling(df: pd.DataFrame, method: str = 'standard',
                 columns: Optional[List[str]] = None,
                 scaler: Any = None) -> Tuple[pd.DataFrame, Any]:
    """
    Apply different scaling methods to numerical features
    
    Args:
        df: Input DataFrame
        method: 'standard', 'robust', 'minmax', 'quantile', 'power'
        columns: Columns to scale (None = all numerical)
        scaler: Existing scaler object (if None, will create and fit)
        
    Returns:
        Tuple of (scaled DataFrame, scaler object)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [c for c in columns if c != 'Target']
    
    if not columns:
        return df.copy(), scaler
    
    df_scaled = df.copy()
    
    if scaler is None:
        # Fit phase
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'quantile':
            scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        elif method == 'power':
            scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        else:
            logger.warning(f"Unknown scaling method: {method}, using StandardScaler")
            scaler = StandardScaler()
        
        df_scaled[columns] = scaler.fit_transform(df[columns])
    else:
        # Transform phase
        # Check if columns match what the scaler expects? 
        # For simplicity, we assume columns provided match training columns or strictly follow order
        try:
            df_scaled[columns] = scaler.transform(df[columns])
        except Exception as e:
            logger.warning(f"Scaling transform failed: {e}. Returning unscaled data for these columns.")
    
    return df_scaled, scaler


def create_interaction_features(df: pd.DataFrame, 
                               feature_pairs: Optional[List[Tuple[str, str]]] = None,
                               max_interactions: int = 10) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """
    Create interaction features (multiplication of feature pairs)
    
    Args:
        df: Input DataFrame
        feature_pairs: List of (col1, col2) tuples (None = auto-generate top correlations from df)
        max_interactions: Maximum number of interactions to create
        
    Returns:
        Tuple (DataFrame with interaction features, feature_pairs used)
    """
    df_enhanced = df.copy()
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Target' in numerical_cols:
        numerical_cols.remove('Target')
    
    if len(numerical_cols) < 2:
        return df_enhanced, []
    
    if feature_pairs is None:
        # Auto-generate: find top correlated pairs (Fit phase)
        corr_matrix = df[numerical_cols].corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        
        feature_pairs_list = []
        for i in range(min(max_interactions, len(numerical_cols) * (len(numerical_cols) - 1) // 2)):
            try:
                max_corr_idx = np.unravel_index(corr_matrix.values.argmax(), corr_matrix.shape)
                col1, col2 = corr_matrix.index[max_corr_idx[0]], corr_matrix.columns[max_corr_idx[1]]
                if (col1, col2) not in feature_pairs_list and (col2, col1) not in feature_pairs_list:
                    feature_pairs_list.append((col1, col2))
                corr_matrix.iloc[max_corr_idx[0], max_corr_idx[1]] = 0
            except ValueError:
                break
        feature_pairs = feature_pairs_list
    
    # Create interaction features
    created_pairs = []
    if feature_pairs:
        for col1, col2 in feature_pairs:
            if col1 in df.columns and col2 in df.columns:
                interaction_name = f"{col1}_x_{col2}"
                df_enhanced[interaction_name] = df[col1] * df[col2]
                created_pairs.append((col1, col2))
    
    return df_enhanced, created_pairs


def create_binned_features(df: pd.DataFrame, columns: Optional[List[str]] = None,
                          n_bins: int = 5, strategy: str = 'quantile',
                          binners: Dict = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Create binned features (Fit/Transform)
    """
    from sklearn.preprocessing import KBinsDiscretizer
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Target' in columns:
            columns.remove('Target')
    
    if not columns:
        return df.copy(), {}
    
    df_enhanced = df.copy()
    if binners is None:
        binners = {}
        # Fit phase
        for col in columns:
            if col not in df.columns:
                continue
            try:
                binner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
                binned_col = f"{col}_binned"
                # Check for low variance causing bin edges to not be unique
                df_enhanced[binned_col] = binner.fit_transform(df[[col]]).flatten()
                binners[col] = binner
            except Exception as e:
                logger.debug(f"Skipping binning for {col}: {e}")
    else:
        # Transform phase
        for col, binner in binners.items():
            if col in df.columns:
                binned_col = f"{col}_binned"
                try:
                    df_enhanced[binned_col] = binner.transform(df[[col]]).flatten()
                except Exception:
                    pass
    
    return df_enhanced, binners


def apply_log_transform(df: pd.DataFrame, columns: Optional[List[str]] = None,
                        add_one: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply log transformation to numerical features.
    If columns is None, detects likely skewed columns (Fit phase).
    If columns is list, applies to those columns (Transform phase).
    """
    df_transformed = df.copy()
    
    if columns is None:
        # Auto-detect skewed columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Target' in numerical_cols:
            numerical_cols.remove('Target')
        
        columns = []
        for col in numerical_cols:
            if col in df.columns:
                # Simple skew check - safe to do on train only and apply to test
                skewness = df[col].skew()
                if abs(skewness) > 1.0:
                    columns.append(col)
    
    for col in columns:
        if col in df.columns:
            log_col = f"{col}_log"
            if add_one:
                df_transformed[log_col] = np.log1p(df[col].clip(lower=0)) # Ensure non-negative
            else:
                df_transformed[log_col] = np.log(df[col].clip(lower=1e-10))
    
    return df_transformed, columns


def remove_multicollinearity(df: pd.DataFrame, threshold: float = 0.95,
                            target_col: str = 'Target',
                            keep_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove highly correlated features.
    Fit phase: Calculates correlation, determines columns to keep.
    Transform phase: Keeps only the columns determined in Fit phase.
    """
    if keep_columns is not None:
        # Transform phase: just select the columns
        # Ensure we don't try to select columns that don't exist (if input is weird)
        # Also ensure we keep Target if it exists
        valid_cols = [c for c in keep_columns if c in df.columns]
        if target_col in df.columns and target_col not in valid_cols:
             valid_cols.append(target_col)
        return df[valid_cols], keep_columns
    
    # Fit phase:
    feature_cols = [c for c in df.columns if c != target_col]
    numerical_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) < 2:
        return df.copy(), df.columns.tolist()
    
    corr_matrix = df[numerical_cols].corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    
    df_cleaned = df.drop(columns=to_drop)
    return df_cleaned, df_cleaned.columns.tolist()


def select_features_statistical(X: pd.DataFrame, y: Optional[pd.Series], 
                               method: str = 'f_classif', k: int = 50,
                               selector: Any = None) -> Tuple[pd.DataFrame, Any]:
    """
    Select features using statistical tests (Fit/Transform)
    """
    if selector is None:
        if y is None:
            raise ValueError("y is required for fitting feature selection")
        # Fit phase
        if method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        elif method == 'mutual_info_classif':
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        X_selected = selector.fit_transform(X, y)
    else:
        # Transform phase
        X_selected = selector.transform(X)
        
    selected_features = X.columns[selector.get_support()].tolist()
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    return X_selected_df, selector


def remove_low_variance_features(df: pd.DataFrame, threshold: float = 0.01,
                                target_col: str = 'Target',
                                selector: Any = None) -> Tuple[pd.DataFrame, Any]:
    """
    Remove low variance features (Fit/Transform)
    """
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    
    if selector is None:
        # Fit phase
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)
    else:
        # Transform phase
        X_selected = selector.transform(X)
    
    selected_features = X.columns[selector.get_support()].tolist()
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=df.index)
    
    if target_col in df.columns:
        X_selected_df[target_col] = df[target_col]
        
    return X_selected_df, selector


def advanced_preprocessing_pipeline(df: pd.DataFrame, 
                                    target_col: str = 'Target',
                                    handle_outliers_flag: bool = True,
                                    outlier_method: str = 'cap',
                                    apply_scaling_flag: bool = False,
                                    scaling_method: str = 'robust',
                                    create_interactions_flag: bool = False,
                                    max_interactions: int = 10,
                                    create_bins_flag: bool = False,
                                    apply_log_transform_flag: bool = False,
                                    remove_multicollinear_flag: bool = True,
                                    remove_low_variance_flag: bool = True,
                                    feature_selection: Optional[str] = None,
                                    n_features_select: int = 50,
                                    artifacts: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Complete advanced preprocessing pipeline with Fit/Transform separation
    
    Args:
        df: Input DataFrame (Train or Test)
        artifacts: Dictionary of artifacts from training (if None, will fit and return new artifacts)
        
    Returns:
        Tuple of (processed DataFrame, artifacts dictionary)
    """
    df_processed = df.copy()
    
    # Initialize artifacts if fit phase
    is_fit = artifacts is None
    if is_fit:
        artifacts = {}
        logger.info("Starting advanced preprocessing pipeline (FIT phase)...")
    else:
        logger.info("Starting advanced preprocessing pipeline (TRANSFORM phase)...")
    
    # Step 1: Handle outliers
    if handle_outliers_flag:
        if is_fit:
            # We assume capping is safer than removing to avoid data loss, or we use bounds
            bounds = None
        else:
            bounds = artifacts.get('outlier_bounds')
            
        df_processed, new_bounds = handle_outliers(df_processed, method=outlier_method, bounds=bounds)
        if is_fit:
            artifacts['outlier_bounds'] = new_bounds

    # Step 2: Log Transform
    if apply_log_transform_flag:
        log_cols = artifacts.get('log_columns') if not is_fit else None
        df_processed, log_cols = apply_log_transform(df_processed, columns=log_cols)
        if is_fit:
            artifacts['log_columns'] = log_cols

    # Step 3: Binning
    if create_bins_flag:
        binners = artifacts.get('binners') if not is_fit else None
        df_processed, binners = create_binned_features(df_processed, binners=binners)
        if is_fit:
            artifacts['binners'] = binners

    # Step 4: Interactions
    if create_interactions_flag:
        pairs = artifacts.get('interaction_pairs') if not is_fit else None
        df_processed, pairs = create_interaction_features(df_processed, feature_pairs=pairs, max_interactions=max_interactions)
        if is_fit:
            artifacts['interaction_pairs'] = pairs

    # Step 5: Low Variance
    if remove_low_variance_flag:
        var_selector = artifacts.get('variance_selector') if not is_fit else None
        try:
            df_processed, var_selector = remove_low_variance_features(df_processed, target_col=target_col, selector=var_selector)
            if is_fit:
                artifacts['variance_selector'] = var_selector
        except Exception as e:
            logger.warning(f"Low variance removal failed: {e}")

    # Step 6: Multicollinearity
    if remove_multicollinear_flag:
        keep_cols = artifacts.get('multicollinear_keep_cols') if not is_fit else None
        df_processed, keep_cols = remove_multicollinearity(df_processed, target_col=target_col, keep_columns=keep_cols)
        if is_fit:
            artifacts['multicollinear_keep_cols'] = keep_cols

    # Step 7: Scaling (Optional here, usually in Pipeline)
    if apply_scaling_flag:
        scaler = artifacts.get('scaler') if not is_fit else None
        df_processed, scaler = apply_scaling(df_processed, method=scaling_method, scaler=scaler)
        if is_fit:
            artifacts['scaler'] = scaler

    # Step 8: Feature Selection
    if feature_selection:
        feat_selector = artifacts.get('feature_selector') if not is_fit else None
        
        # Determine X and y
        y = None
        if target_col in df_processed.columns:
            y = df_processed[target_col]
            X = df_processed.drop(columns=[target_col])
        else:
            X = df_processed # Test set might not have target or it was dropped
            
        try:
            if is_fit: # Needs y
                if y is not None:
                     X_sel, feat_selector = select_features_statistical(X, y, method=feature_selection, k=n_features_select, selector=None)
                     if target_col in df_processed.columns:
                         X_sel[target_col] = y
                     df_processed = X_sel
                     artifacts['feature_selector'] = feat_selector
            else:
                 if feat_selector:
                     X_sel, _ = select_features_statistical(X, None, selector=feat_selector)
                     if target_col in df_processed.columns:
                         X_sel[target_col] = y
                     df_processed = X_sel
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}")

    logger.info(f"Advanced preprocessing completed. Shape: {df_processed.shape}")
    return df_processed, artifacts



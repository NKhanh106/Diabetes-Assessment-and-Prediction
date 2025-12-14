"""
Advanced preprocessing techniques for maximum model performance
Includes outlier detection, feature engineering, scaling options, and more
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
                       factor: float = 1.5) -> Dict[str, List[int]]:
    """
    Detect outliers using Interquartile Range (IQR) method
    
    Args:
        df: Input DataFrame
        columns: List of columns to check (None = all numerical)
        factor: IQR factor (default 1.5)
        
    Returns:
        Dictionary mapping column names to outlier indices
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers = {}
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_indices = df[outlier_mask].index.tolist()
        
        if outlier_indices:
            outliers[col] = outlier_indices
    
    return outliers


def detect_outliers_zscore(df: pd.DataFrame, columns: Optional[List[str]] = None,
                           threshold: float = 3.0) -> Dict[str, List[int]]:
    """
    Detect outliers using Z-score method
    
    Args:
        df: Input DataFrame
        columns: List of columns to check (None = all numerical)
        threshold: Z-score threshold (default 3.0)
        
    Returns:
        Dictionary mapping column names to outlier indices
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers = {}
    for col in columns:
        if col not in df.columns:
            continue
        
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outlier_mask = z_scores > threshold
        outlier_indices = df[outlier_mask].index.tolist()
        
        if outlier_indices:
            outliers[col] = outlier_indices
    
    return outliers


def handle_outliers(df: pd.DataFrame, method: str = 'cap', 
                   columns: Optional[List[str]] = None,
                   outlier_dict: Optional[Dict[str, List[int]]] = None) -> pd.DataFrame:
    """
    Handle outliers in numerical columns
    
    Args:
        df: Input DataFrame
        method: 'cap' (cap at bounds), 'remove' (remove rows), 'winsorize' (winsorize)
        columns: Columns to process (None = all numerical)
        outlier_dict: Pre-computed outliers (if None, will detect)
        
    Returns:
        DataFrame with outliers handled
    """
    df_processed = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if outlier_dict is None:
        outlier_dict = detect_outliers_iqr(df, columns)
    
    if method == 'cap':
        # Cap outliers at IQR bounds
        for col in columns:
            if col not in df.columns:
                continue
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
            
    elif method == 'winsorize':
        # Winsorize at 1st and 99th percentiles
        for col in columns:
            if col not in df.columns:
                continue
                
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df_processed[col] = df_processed[col].clip(lower=lower, upper=upper)
            
    elif method == 'remove':
        # Remove rows with outliers
        outlier_indices = set()
        for indices in outlier_dict.values():
            outlier_indices.update(indices)
        df_processed = df_processed.drop(index=list(outlier_indices))
        
    return df_processed


def apply_scaling(df: pd.DataFrame, method: str = 'standard',
                 columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Any]:
    """
    Apply different scaling methods to numerical features
    
    Args:
        df: Input DataFrame
        method: 'standard', 'robust', 'minmax', 'quantile', 'power'
        columns: Columns to scale (None = all numerical)
        
    Returns:
        Tuple of (scaled DataFrame, scaler object)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude target if present
        columns = [c for c in columns if c != 'Target']
    
    if not columns:
        return df.copy(), None
    
    df_scaled = df.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()  # Better for outliers
    elif method == 'minmax':
        scaler = MinMaxScaler()  # Scale to [0, 1]
    elif method == 'quantile':
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    elif method == 'power':
        scaler = PowerTransformer(method='yeo-johnson', standardize=True)
    else:
        logger.warning(f"Unknown scaling method: {method}, using StandardScaler")
        scaler = StandardScaler()
    
    df_scaled[columns] = scaler.fit_transform(df[columns])
    
    return df_scaled, scaler


def create_interaction_features(df: pd.DataFrame, 
                               feature_pairs: Optional[List[Tuple[str, str]]] = None,
                               max_interactions: int = 10) -> pd.DataFrame:
    """
    Create interaction features (multiplication of feature pairs)
    
    Args:
        df: Input DataFrame
        feature_pairs: List of (col1, col2) tuples (None = auto-generate top correlations)
        max_interactions: Maximum number of interactions to create
        
    Returns:
        DataFrame with interaction features added
    """
    df_enhanced = df.copy()
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Target' in numerical_cols:
        numerical_cols.remove('Target')
    
    if len(numerical_cols) < 2:
        return df_enhanced
    
    if feature_pairs is None:
        # Auto-generate: find top correlated pairs
        corr_matrix = df[numerical_cols].corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)  # Remove diagonal
        
        # Get top correlations
        feature_pairs = []
        for i in range(min(max_interactions, len(numerical_cols) * (len(numerical_cols) - 1) // 2)):
            max_corr_idx = np.unravel_index(corr_matrix.values.argmax(), corr_matrix.shape)
            col1, col2 = corr_matrix.index[max_corr_idx[0]], corr_matrix.columns[max_corr_idx[1]]
            if (col1, col2) not in feature_pairs and (col2, col1) not in feature_pairs:
                feature_pairs.append((col1, col2))
            corr_matrix.iloc[max_corr_idx[0], max_corr_idx[1]] = 0
    
    # Create interaction features
    interaction_count = 0
    for col1, col2 in feature_pairs:
        if col1 in df.columns and col2 in df.columns:
            interaction_name = f"{col1}_x_{col2}"
            if interaction_name not in df_enhanced.columns:
                df_enhanced[interaction_name] = df[col1] * df[col2]
                interaction_count += 1
                if interaction_count >= max_interactions:
                    break
    
    logger.info(f"Created {interaction_count} interaction features")
    return df_enhanced


def create_binned_features(df: pd.DataFrame, columns: Optional[List[str]] = None,
                          n_bins: int = 5, strategy: str = 'quantile') -> pd.DataFrame:
    """
    Create binned (discretized) features for numerical columns
    
    Args:
        df: Input DataFrame
        columns: Columns to bin (None = all numerical)
        n_bins: Number of bins
        strategy: 'quantile' or 'uniform'
        
    Returns:
        DataFrame with binned features added
    """
    from sklearn.preprocessing import KBinsDiscretizer
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Target' in columns:
            columns.remove('Target')
    
    if not columns:
        return df.copy()
    
    df_enhanced = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        binner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        binned_col = f"{col}_binned"
        df_enhanced[binned_col] = binner.fit_transform(df[[col]]).flatten()
    
    logger.info(f"Created binned features for {len(columns)} columns")
    return df_enhanced


def apply_log_transform(df: pd.DataFrame, columns: Optional[List[str]] = None,
                        add_one: bool = True) -> pd.DataFrame:
    """
    Apply log transformation to numerical features (useful for skewed data)
    
    Args:
        df: Input DataFrame
        columns: Columns to transform (None = auto-detect skewed columns)
        add_one: Add 1 before log to handle zeros
        
    Returns:
        DataFrame with log-transformed features
    """
    if columns is None:
        # Auto-detect skewed columns (skewness > 1)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Target' in numerical_cols:
            numerical_cols.remove('Target')
        
        columns = []
        for col in numerical_cols:
            if col in df.columns:
                skewness = df[col].skew()
                if abs(skewness) > 1.0:  # Highly skewed
                    columns.append(col)
    
    if not columns:
        return df.copy()
    
    df_transformed = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        log_col = f"{col}_log"
        if add_one:
            df_transformed[log_col] = np.log1p(df[col])
        else:
            df_transformed[log_col] = np.log(df[col] + 1e-10)  # Small epsilon to avoid log(0)
    
    logger.info(f"Applied log transformation to {len(columns)} columns")
    return df_transformed


def remove_multicollinearity(df: pd.DataFrame, threshold: float = 0.95,
                            target_col: str = 'Target') -> pd.DataFrame:
    """
    Remove highly correlated features to reduce multicollinearity
    
    Args:
        df: Input DataFrame
        threshold: Correlation threshold (default 0.95)
        target_col: Target column name to exclude
        
    Returns:
        DataFrame with multicollinear features removed
    """
    feature_cols = [c for c in df.columns if c != target_col]
    numerical_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) < 2:
        return df.copy()
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr().abs()
    
    # Find pairs with high correlation
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features to drop
    to_drop = [column for column in upper_triangle.columns 
               if any(upper_triangle[column] > threshold)]
    
    if to_drop:
        logger.info(f"Removing {len(to_drop)} multicollinear features: {to_drop[:5]}...")
        df_cleaned = df.drop(columns=to_drop)
    else:
        df_cleaned = df.copy()
    
    return df_cleaned


def select_features_statistical(X: pd.DataFrame, y: pd.Series, 
                               method: str = 'f_classif', k: int = 50) -> Tuple[pd.DataFrame, Any]:
    """
    Select top features using statistical methods
    
    Args:
        X: Feature DataFrame
        y: Target Series
        method: 'f_classif' or 'mutual_info_classif'
        k: Number of features to select
        
    Returns:
        Tuple of (selected features DataFrame, selector object)
    """
    if method == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    elif method == 'mutual_info_classif':
        selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    logger.info(f"Selected {len(selected_features)} features using {method}")
    return X_selected_df, selector


def remove_low_variance_features(df: pd.DataFrame, threshold: float = 0.01,
                                target_col: str = 'Target') -> Tuple[pd.DataFrame, Any]:
    """
    Remove features with low variance (likely uninformative)
    
    Args:
        df: Input DataFrame
        threshold: Variance threshold
        target_col: Target column name to exclude
        
    Returns:
        Tuple of (cleaned DataFrame, VarianceThreshold object)
    """
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    
    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(X)
    
    selected_features = X.columns[selector.get_support()].tolist()
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=df.index)
    
    # Add target back if present
    if target_col in df.columns:
        X_selected_df[target_col] = df[target_col]
    
    removed_count = len(feature_cols) - len(selected_features)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} low-variance features")
    
    return X_selected_df, selector


def advanced_preprocessing_pipeline(df: pd.DataFrame, 
                                    target_col: str = 'Target',
                                    handle_outliers: bool = True,
                                    outlier_method: str = 'cap',
                                    apply_scaling: bool = False,
                                    scaling_method: str = 'robust',
                                    create_interactions: bool = False,
                                    max_interactions: int = 10,
                                    create_bins: bool = False,
                                    apply_log_transform: bool = False,
                                    remove_multicollinear: bool = True,
                                    remove_low_variance: bool = True,
                                    feature_selection: Optional[str] = None,
                                    n_features_select: int = 50) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Complete advanced preprocessing pipeline
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        handle_outliers: Whether to handle outliers
        outlier_method: 'cap', 'winsorize', or 'remove'
        apply_scaling: Whether to apply scaling
        scaling_method: 'standard', 'robust', 'minmax', 'quantile', 'power'
        create_interactions: Whether to create interaction features
        max_interactions: Maximum number of interactions
        create_bins: Whether to create binned features
        apply_log_transform: Whether to apply log transformation
        remove_multicollinear: Whether to remove multicollinear features
        remove_low_variance: Whether to remove low variance features
        feature_selection: 'f_classif', 'mutual_info_classif', or None
        n_features_select: Number of features to select
        
    Returns:
        Tuple of (processed DataFrame, preprocessing artifacts dictionary)
    """
    df_processed = df.copy()
    artifacts = {}
    
    logger.info("Starting advanced preprocessing pipeline...")
    
    # Step 1: Handle outliers
    if handle_outliers:
        logger.info(f"Handling outliers using method: {outlier_method}")
        outliers = detect_outliers_iqr(df_processed, columns=None)
        if outliers:
            logger.info(f"Found outliers in {len(outliers)} columns")
            df_processed = handle_outliers(df_processed, method=outlier_method)
            artifacts['outliers'] = outliers
        else:
            logger.info("No outliers detected")
    
    # Step 2: Feature engineering - Log transformation
    if apply_log_transform:
        logger.info("Applying log transformation to skewed features...")
        df_processed = apply_log_transform(df_processed)
    
    # Step 3: Feature engineering - Binning
    if create_bins:
        logger.info("Creating binned features...")
        df_processed = create_binned_features(df_processed)
    
    # Step 4: Feature engineering - Interactions
    if create_interactions:
        logger.info("Creating interaction features...")
        df_processed = create_interaction_features(df_processed, max_interactions=max_interactions)
    
    # Step 5: Remove low variance features
    if remove_low_variance:
        logger.info("Removing low variance features...")
        df_processed, variance_selector = remove_low_variance_features(df_processed, target_col=target_col)
        artifacts['variance_selector'] = variance_selector
    
    # Step 6: Remove multicollinearity
    if remove_multicollinear:
        logger.info("Removing multicollinear features...")
        df_processed = remove_multicollinearity(df_processed, target_col=target_col)
    
    # Step 7: Apply scaling (if needed - usually done in model pipeline)
    if apply_scaling:
        logger.info(f"Applying {scaling_method} scaling...")
        df_processed, scaler = apply_scaling(df_processed, method=scaling_method)
        artifacts['scaler'] = scaler
    
    # Step 8: Feature selection (if target is available)
    if feature_selection and target_col in df_processed.columns:
        logger.info(f"Selecting features using {feature_selection}...")
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
        X_selected, selector = select_features_statistical(X, y, method=feature_selection, k=n_features_select)
        X_selected[target_col] = y
        df_processed = X_selected
        artifacts['feature_selector'] = selector
    
    logger.info(f"Advanced preprocessing completed. Final shape: {df_processed.shape}")
    return df_processed, artifacts


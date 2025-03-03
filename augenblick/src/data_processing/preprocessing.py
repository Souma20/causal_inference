import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    confounder_cols: List[str],
    categorical_cols: Optional[List[str]] = None,
    scale_numeric: bool = True,
    handle_missing: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Preprocess data for causal inference analysis.
    
    Args:
        data: Input DataFrame
        treatment_col: Name of the treatment column
        outcome_col: Name of the outcome column
        confounder_cols: List of confounding variable columns
        categorical_cols: List of categorical columns requiring encoding
        scale_numeric: Whether to standardize numeric features
        handle_missing: Whether to impute missing values
        
    Returns:
        Tuple of (preprocessed_data, preprocessing_info)
        where preprocessing_info contains the fitted preprocessing objects
    """
    if categorical_cols is None:
        categorical_cols = []
    
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Track preprocessing steps for later use (e.g., for new data)
    preprocessing_info = {}
    
    # Handle missing values if requested
    if handle_missing:
        numeric_cols = [col for col in confounder_cols if col not in categorical_cols]
        
        # Impute numeric features
        if numeric_cols:
            numeric_imputer = SimpleImputer(strategy='mean')
            df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
            preprocessing_info['numeric_imputer'] = numeric_imputer
        
        # Impute categorical features
        if categorical_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
            preprocessing_info['categorical_imputer'] = cat_imputer
    
    # Encode categorical variables
    if categorical_cols:
        encoder = OneHotEncoder(sparse=False, drop='first')
        encoded_cats = encoder.fit_transform(df[categorical_cols])
        
        # Create DataFrame with encoded values
        encoded_df = pd.DataFrame(
            encoded_cats,
            columns=encoder.get_feature_names_out(categorical_cols),
            index=df.index
        )
        
        # Drop original categorical columns and join encoded ones
        df = df.drop(columns=categorical_cols).join(encoded_df)
        
        preprocessing_info['encoder'] = encoder
    
    # Scale numeric features
    if scale_numeric:
        numeric_cols = [col for col in confounder_cols if col not in categorical_cols]
        if numeric_cols:
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            preprocessing_info['scaler'] = scaler
    
    # Ensure treatment and outcome columns are preserved
    if treatment_col not in df.columns:
        raise ValueError(f"Treatment column '{treatment_col}' not found after preprocessing")
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found after preprocessing")
    
    return df, preprocessing_info

def identify_confounders(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    threshold: float = 0.1
) -> List[str]:
    """
    Identify potential confounders by finding variables correlated with both treatment and outcome.
    
    Args:
        data: Input DataFrame
        treatment_col: Name of the treatment column
        outcome_col: Name of the outcome column
        threshold: Correlation threshold to consider a variable as a potential confounder
        
    Returns:
        List of potential confounder column names
    """
    # Get correlation with treatment and outcome
    treatment_corr = data.drop(columns=[treatment_col, outcome_col]).corrwith(data[treatment_col]).abs()
    outcome_corr = data.drop(columns=[treatment_col, outcome_col]).corrwith(data[outcome_col]).abs()
    
    # Find variables correlated with both treatment and outcome
    potential_confounders = []
    for col in treatment_corr.index:
        if treatment_corr[col] > threshold and outcome_corr[col] > threshold:
            potential_confounders.append(col)
    
    return potential_confounders

def balance_check(
    data: pd.DataFrame,
    treatment_col: str,
    confounder_cols: List[str]
) -> pd.DataFrame:
    """
    Check balance of confounders between treatment and control groups.
    
    Args:
        data: Input DataFrame
        treatment_col: Name of the treatment column
        confounder_cols: List of confounding variable columns
        
    Returns:
        DataFrame with balance statistics
    """
    balance_stats = []
    
    # Split data into treatment and control groups
    treatment_group = data[data[treatment_col] == 1]
    control_group = data[data[treatment_col] == 0]
    
    for col in confounder_cols:
        # Calculate statistics
        t_mean = treatment_group[col].mean()
        c_mean = control_group[col].mean()
        t_std = treatment_group[col].std()
        c_std = control_group[col].std()
        
        # Calculate standardized mean difference
        std_pooled = np.sqrt((t_std**2 + c_std**2) / 2)
        std_mean_diff = (t_mean - c_mean) / std_pooled if std_pooled != 0 else np.nan
        
        balance_stats.append({
            'variable': col,
            'treatment_mean': t_mean,
            'control_mean': c_mean,
            'treatment_std': t_std,
            'control_std': c_std,
            'std_mean_diff': std_mean_diff,
            'abs_std_mean_diff': abs(std_mean_diff)
        })
    
    balance_df = pd.DataFrame(balance_stats)
    
    # Sort by absolute standardized mean difference
    return balance_df.sort_values('abs_std_mean_diff', ascending=False)

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Optional, Union

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a dataset from a file path.
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        Loaded dataset as a pandas DataFrame
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
        
    # Determine file type and load accordingly
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.csv':
        return pd.read_csv(file_path)
    elif file_extension == '.xlsx' or file_extension == '.xls':
        return pd.read_excel(file_path)
    elif file_extension == '.json':
        return pd.read_json(file_path)
    elif file_extension == '.parquet':
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def generate_synthetic_data(
    n_samples: int = 1000, 
    binary_treatment: bool = True,
    confounders: Dict[str, Dict] = None,
    treatment_effect: float = 2.0,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic data for causal inference analysis.
    
    Args:
        n_samples: Number of samples to generate
        binary_treatment: Whether treatment is binary (0/1) or continuous
        confounders: Dictionary mapping confounder names to their distribution parameters
                     Example: {'age': {'mean': 50, 'std': 10}, 'bmi': {'mean': 25, 'std': 3}}
        treatment_effect: The true effect of treatment on outcome
        random_state: Random seed for reproducibility
        
    Returns:
        Synthetic dataset as a pandas DataFrame
    """
    np.random.seed(random_state)
    
    # Default confounders if none provided
    if confounders is None:
        confounders = {
            'age': {'mean': 50, 'std': 10},
            'blood_pressure': {'mean': 120, 'std': 10},
            'bmi': {'mean': 25, 'std': 3}
        }
    
    # Generate confounders
    data = {}
    for name, params in confounders.items():
        if 'mean' in params and 'std' in params:
            data[name] = np.random.normal(params['mean'], params['std'], n_samples)
        elif 'low' in params and 'high' in params:
            data[name] = np.random.uniform(params['low'], params['high'], n_samples)
        else:
            raise ValueError(f"Invalid parameters for confounder {name}")
    
    # Generate propensity scores based on confounders
    propensity = 0.3
    for name in confounders:
        # Normalize to [0, 1] range and add some weight
        propensity += 0.1 * (data[name] - np.min(data[name])) / (np.max(data[name]) - np.min(data[name]))
    
    propensity = 1 / (1 + np.exp(-propensity))  # Sigmoid function to bound between 0 and 1
    
    # Generate treatment
    if binary_treatment:
        data['treatment'] = (np.random.random(n_samples) < propensity).astype(int)
    else:
        data['treatment'] = propensity + np.random.normal(0, 0.1, n_samples)
        data['treatment'] = np.clip(data['treatment'], 0, 1)  # Bound between 0 and 1
    
    # Generate outcome with treatment effect
    outcome = 0.5
    for name in confounders:
        # Add confounder effect on outcome
        outcome += 0.05 * data[name]
    
    # Add treatment effect
    outcome += treatment_effect * data['treatment']
    
    # Add noise
    outcome += np.random.normal(0, 1, n_samples)
    
    data['outcome'] = outcome
    
    return pd.DataFrame(data)

def split_data(
    data: pd.DataFrame, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataset into training and test sets.
    
    Args:
        data: Input DataFrame
        test_size: Proportion of the dataset to include in the test split
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (training_data, test_data)
    """
    from sklearn.model_selection import train_test_split
    
    return train_test_split(data, test_size=test_size, random_state=random_state)

def get_example_dataset() -> pd.DataFrame:
    """
    Get a simple example dataset for testing purposes.
    
    Returns:
        Example dataset as a pandas DataFrame
    """
    return generate_synthetic_data(n_samples=500)

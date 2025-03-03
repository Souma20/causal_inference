import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import io, base64


def load_ihdp_data(csv_path):
    """
    Loads the IHDP dataset from a CSV file.
    Expected columns:
      - 'treatment': Treatment assignment (0/1)
      - 'y_factual': Observed outcome
      - Covariates: 'x1' ... 'x25'
    """
    print(f"Loading data from {csv_path}")  # Debug log
    df = pd.read_csv(csv_path)
    print(f"Data loaded successfully. Shape: {df.shape}")  # Debug log
    print("Columns:", df.columns.tolist())  # Debug log
    print("Sample of treatment values:", df['treatment'].value_counts())  # Debug log
    return df

def calculate_naive_ate(df):
    """Calculate naive ATE by simple difference in means"""
    treated = df[df['treatment'] == 1]['y_factual'].mean()
    control = df[df['treatment'] == 0]['y_factual'].mean()
    return float(treated - control)

def calculate_propensity_scores(df):
    """Calculate propensity scores using logistic regression"""
    covariates = [col for col in df.columns if col.startswith('x')]
    X = df[covariates]
    treatment = df['treatment']
    
    ps_model = LogisticRegression(random_state=42)
    ps_model.fit(X, treatment)
    return ps_model.predict_proba(X)[:, 1]

def match_samples(df, ps_scores, n_neighbors=1):
    """Match treated and control units based on propensity scores"""
    treated_idx = df[df['treatment'] == 1].index
    control_idx = df[df['treatment'] == 0].index
    
    treated_ps = ps_scores[treated_idx].reshape(-1, 1)
    control_ps = ps_scores[control_idx].reshape(-1, 1)
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(control_ps)
    
    distances, indices = nbrs.kneighbors(treated_ps)
    matched_control_idx = control_idx[indices.flatten()]
    
    return treated_idx, matched_control_idx

def calculate_psm_ate(df, num_neighbors=5):
    """Calculate ATE using Propensity Score Matching"""
    ps_scores = calculate_propensity_scores(df)
    treated_idx, matched_control_idx = match_samples(df, ps_scores, n_neighbors=num_neighbors)
    
    treated_outcomes = df.loc[treated_idx, 'y_factual'].mean()
    control_outcomes = df.loc[matched_control_idx, 'y_factual'].mean()
    
    return float(treated_outcomes - control_outcomes)

def calculate_ipw_ate(df, trim_weights=0.1):
    """Calculate ATE using Inverse Probability Weighting"""
    ps_scores = calculate_propensity_scores(df)
    
    # Trim extreme propensity scores based on the trim_weights parameter
    ps_scores = np.clip(ps_scores, trim_weights, 1 - trim_weights)
    
    # Calculate weights
    weights = df['treatment'] / ps_scores + (1 - df['treatment']) / (1 - ps_scores)
    
    # Calculate weighted outcomes
    weighted_treated = (df['treatment'] * df['y_factual'] / ps_scores).sum() / (df['treatment'] / ps_scores).sum()
    weighted_control = ((1 - df['treatment']) * df['y_factual'] / (1 - ps_scores)).sum() / ((1 - df['treatment']) / (1 - ps_scores)).sum()
    
    return float(weighted_treated - weighted_control)

def create_balance_plot(df, ps_scores):
    """Create covariate balance plot"""
    print("Starting to create balance plot...")  # Debug log
    
    # Clear any existing plots
    plt.clf()
    
    # Create figure with specific size
    plt.figure(figsize=(12, 8))
    
    # Get covariates and sort them for better visualization
    covariates = sorted([col for col in df.columns if col.startswith('x')])
    print(f"Found {len(covariates)} covariates")  # Debug log
    
    # Calculate standardized differences before matching
    std_diff = []
    for cov in covariates:
        treated_mean = df[df['treatment'] == 1][cov].mean()
        treated_var = df[df['treatment'] == 1][cov].var()
        control_mean = df[df['treatment'] == 0][cov].mean()
        control_var = df[df['treatment'] == 0][cov].var()
        
        pooled_sd = np.sqrt((treated_var + control_var) / 2)
        if pooled_sd == 0:  # Handle division by zero
            std_diff.append(0)
        else:
            std_diff.append((treated_mean - control_mean) / pooled_sd)
    
    print(f"Calculated {len(std_diff)} standardized differences")  # Debug log
    print("Standardized differences:", std_diff)  # Debug log
    
    # Create horizontal bar plot with specific style
    bars = plt.barh(covariates, std_diff, height=0.6, color='skyblue', alpha=0.6)
    
    # Add vertical line at x=0
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Customize plot appearance
    plt.title('Covariate Balance Plot', fontsize=14, pad=20)
    plt.xlabel('Standardized Difference', fontsize=12)
    plt.ylabel('Covariates', fontsize=12)
    
    # Add grid and adjust style
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    # Adjust layout to prevent text cutoff
    plt.tight_layout()
    
    # Convert plot to base64 string with high DPI for better quality
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, 
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close()
    
    base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    print(f"Generated base64 string of length: {len(base64_str)}")  # Debug log
    return base64_str

def run_analysis(csv_path, method='PSM', num_neighbors=5, trim_weights=0.1):
    """
    Run causal inference analysis using specified method
    Returns estimated ATE and visualization data
    """
    df = load_ihdp_data(csv_path)
    
    results = {
        'method': method,
        'data': {
            'labels': ['Control', 'Treated'],
            'outcomes': [
                float(df[df['treatment'] == 0]['y_factual'].mean()),
                float(df[df['treatment'] == 1]['y_factual'].mean())
            ]
        }
    }
    
    # Calculate ATEs using different methods
    results['naive_ate'] = calculate_naive_ate(df)
    results['psm_ate'] = calculate_psm_ate(df, num_neighbors=num_neighbors)
    results['ipw_ate'] = calculate_ipw_ate(df, trim_weights=trim_weights)
    
    # Add propensity score distribution
    ps_scores = calculate_propensity_scores(df)
    results['propensity_scores'] = {
        'treated': ps_scores[df['treatment'] == 1].tolist(),
        'control': ps_scores[df['treatment'] == 0].tolist()
    }
    
    # Add covariate balance plot
    results['balance_plot'] = create_balance_plot(df, ps_scores)
    
    # Add covariate names and standardized differences for the balance chart
    covariates = [col for col in df.columns if col.startswith('x')]
    results['covariateLabels'] = covariates
    
    return results

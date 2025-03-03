import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from src.data_processing.data_loader import generate_synthetic_data
from src.models.propensity_score import PropensityScoreModel

def plot_outcome_by_treatment(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    confounder_col: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot outcome by treatment, optionally stratified by a confounder.
    
    Args:
        data: Input DataFrame
        treatment_col: Name of the treatment column
        outcome_col: Name of the outcome column
        confounder_col: Optional confounder column to stratify by
        figsize: Size of the figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if confounder_col is None:
        # Simple boxplot of outcome by treatment
        sns.boxplot(x=treatment_col, y=outcome_col, data=data, ax=ax)
        
        # Add swarmplot for individual points
        sns.swarmplot(x=treatment_col, y=outcome_col, data=data, color='0.25', alpha=0.5, ax=ax)
        
        # Add title and labels
        plt.title(f'Distribution of {outcome_col} by {treatment_col}')
        plt.xlabel(treatment_col)
        plt.ylabel(outcome_col)
    else:
        # Boxplot stratified by confounder
        sns.boxplot(x=treatment_col, y=outcome_col, hue=confounder_col, data=data, ax=ax)
        
        # Add title and labels
        plt.title(f'Distribution of {outcome_col} by {treatment_col}, stratified by {confounder_col}')
        plt.xlabel(treatment_col)
        plt.ylabel(outcome_col)
        plt.legend(title=confounder_col)
    
    return fig

def plot_confounder_distribution(
    data: pd.DataFrame,
    treatment_col: str,
    confounders: List[str],
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot the distribution of confounders by treatment group.
    
    Args:
        data: Input DataFrame
        treatment_col: Name of the treatment column
        confounders: List of confounding variables to plot
        figsize: Size of the figure
        
    Returns:
        Matplotlib figure
    """
    n_confounders = len(confounders)
    n_rows = (n_confounders + 1) // 2  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, confounder in enumerate(confounders):
        if i < len(axes):
            # Check if confounder is numeric or categorical
            if pd.api.types.is_numeric_dtype(data[confounder]):
                # For numeric confounders, use histograms
                sns.histplot(
                    data=data, 
                    x=confounder, 
                    hue=treatment_col,
                    bins=20,
                    alpha=0.5,
                    ax=axes[i]
                )
            else:
                # For categorical confounders, use countplots
                sns.countplot(
                    data=data,
                    x=confounder,
                    hue=treatment_col,
                    ax=axes[i]
                )
                axes[i].tick_params(axis='x', rotation=45)
            
            axes[i].set_title(f'Distribution of {confounder} by {treatment_col}')
    
    # Hide any unused subplots
    for j in range(n_confounders, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_treatment_effect(
    effect_estimates: Dict[str, Dict[str, float]],
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot treatment effect estimates from multiple models.
    
    Args:
        effect_estimates: Dictionary mapping model names to their effect estimates
                         Each effect estimate should have 'ate', 'ci_lower', and 'ci_upper' keys
        figsize: Size of the figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    models = list(effect_estimates.keys())
    ates = [effect_estimates[model]['ate'] for model in models]
    ci_lowers = [effect_estimates[model]['ci_lower'] for model in models]
    ci_uppers = [effect_estimates[model]['ci_upper'] for model in models]
    
    # Calculate error bar lengths
    yerr_lower = [ate - ci_lower for ate, ci_lower in zip(ates, ci_lowers)]
    yerr_upper = [ci_upper - ate for ate, ci_upper in zip(ates, ci_uppers)]
    
    # Plot point estimates and confidence intervals
    ax.errorbar(
        x=models,
        y=ates,
        yerr=[yerr_lower, yerr_upper],
        fmt='o',
        capsize=5,
        elinewidth=2,
        markeredgewidth=2
    )
    
    # Add reference line at zero
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add labels and title
    ax.set_xlabel('Model')
    ax.set_ylabel('Average Treatment Effect')
    ax.set_title('Treatment Effect Estimates with 95% Confidence Intervals')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_covariate_balance(
    balance_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot covariate balance before and after adjustment.
    
    Args:
        balance_df: DataFrame with balance statistics
                   Should have columns 'variable', 'std_mean_diff_before', and 'std_mean_diff_after'
        figsize: Size of the figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort variables by standardized mean difference before adjustment
    sorted_df = balance_df.sort_values('std_mean_diff_before', key=abs, ascending=False)
    
    # Plot
    variables = sorted_df['variable'].tolist()
    before_vals = sorted_df['std_mean_diff_before'].tolist()
    after_vals = sorted_df['std_mean_diff_after'].tolist()
    
    # Set positions for variables
    y_pos = range(len(variables))
    
    # Plot horizontal lines for before and after
    ax.hlines(y=y_pos, xmin=0, xmax=before_vals, color='red', alpha=0.5, linewidth=2, label='Before')
    ax.hlines(y=y_pos, xmin=0, xmax=after_vals, color='blue', alpha=0.5, linewidth=2, label='After')
    
    # Add points at the end of each line
    ax.scatter(before_vals, y_pos, color='red', s=50, alpha=0.5)
    ax.scatter(after_vals, y_pos, color='blue', s=50, alpha=0.5)
    
    # Add vertical reference line at 0
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add reference lines for thresholds of balance
    ax.axvline(x=0.1, color='green', linestyle='--', alpha=0.3)
    ax.axvline(x=-0.1, color='green', linestyle='--', alpha=0.3)
    
    # Add labels and title
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables)
    ax.set_xlabel('Standardized Mean Difference')
    ax.set_title('Covariate Balance Before and After Adjustment')
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_interactive_balance(balance_df: pd.DataFrame) -> go.Figure:
    """
    Create an interactive plot of covariate balance using Plotly.
    
    Args:
        balance_df: DataFrame with balance statistics
                   Should have columns 'variable', 'std_mean_diff_before', and 'std_mean_diff_after'
        
    Returns:
        Plotly figure
    """
    # Sort variables by absolute standardized mean difference before adjustment
    sorted_df = balance_df.sort_values('std_mean_diff_before', key=abs, ascending=False)
    
    # Create figure
    fig = go.Figure()
    
    # Add before and after traces
    fig.add_trace(go.Bar(
        x=sorted_df['std_mean_diff_before'],
        y=sorted_df['variable'],
        name='Before Adjustment',
        orientation='h',
        marker_color='rgba(255, 0, 0, 0.6)'
    ))
    
    fig.add_trace(go.Bar(
        x=sorted_df['std_mean_diff_after'],
        y=sorted_df['variable'],
        name='After Adjustment',
        orientation='h',
        marker_color='rgba(0, 0, 255, 0.6)'
    ))
    
    # Add reference lines
    fig.add_shape(
        type="line",
        x0=0, y0=-0.5,
        x1=0, y1=len(sorted_df) - 0.5,
        line=dict(color="black", width=1, dash="solid")
    )
    
    fig.add_shape(
        type="line",
        x0=0.1, y0=-0.5,
        x1=0.1, y1=len(sorted_df) - 0.5,
        line=dict(color="green", width=1, dash="dash")
    )
    
    fig.add_shape(
        type="line",
        x0=-0.1, y0=-0.5,
        x1=-0.1, y1=len(sorted_df) - 0.5,
        line=dict(color="green", width=1, dash="dash")
    )
    
    # Update layout
    fig.update_layout(
        title="Covariate Balance Before and After Adjustment",
        xaxis_title="Standardized Mean Difference",
        yaxis_title="Variables",
        barmode='group',
        height=600,
        width=900,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_interactive_propensity(
    data: pd.DataFrame,
    treatment_col: str,
    propensity_col: str = 'propensity_score'
) -> go.Figure:
    """
    Create an interactive histogram of propensity scores using Plotly.
    
    Args:
        data: Input DataFrame with propensity scores
        treatment_col: Name of the treatment column
        propensity_col: Name of the propensity score column
        
    Returns:
        Plotly figure
    """
    # Create figure
    fig = go.Figure()
    
    # Add histograms for treatment and control groups
    for treatment_value, color in [(1, 'blue'), (0, 'red')]:
        group_data = data[data[treatment_col] == treatment_value]
        
        fig.add_trace(go.Histogram(
            x=group_data[propensity_col],
            opacity=0.6,
            name=f"{treatment_col} = {treatment_value}",
            marker_color=color,
            nbinsx=30
        ))
    
    # Update layout
    fig.update_layout(
        title="Propensity Score Distribution by Treatment Group",
        xaxis_title="Propensity Score",
        yaxis_title="Count",
        barmode='overlay',
        height=500,
        width=800
    )
    
    return fig

# Generate data
data = generate_synthetic_data(n_samples=1000)

# Run analysis
psm = PropensityScoreModel()
psm.fit(data, 'treatment', ['age', 'blood_pressure', 'bmi'])
matched_data = psm.match()
effect = psm.estimate_effect('outcome')
print(f"Estimated effect: {effect['ate']}")

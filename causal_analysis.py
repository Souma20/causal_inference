#causal_analysis.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import streamlit as st  # For displaying validation metrics in the dashboard

def load_data(filepath):
    """
    Loads the dataset from a CSV file and converts the treatment column.
    """
    df = pd.read_csv(filepath)
    df['treatment'] = df['treatment'].astype(int)
    return df

def load_ihdp_data(filepath):
    """
    Loads and preprocesses the IHDP dataset.
    """
    df = pd.read_csv(filepath)
    df['treatment'] = df['treatment'].astype(int)
    column_descriptions = {
        'treatment': 'Specialized childcare intervention',
        'y_factual': 'Observed cognitive test score',
        'mu0': 'Potential outcome without treatment',
        'mu1': 'Potential outcome with treatment'
    }
    return df, column_descriptions

def plot_outcome_distribution(df):
    """
    Plots the distribution of the factual outcome.
    """
    sns.histplot(df['y_factual'], kde=True)
    plt.title("Distribution of y_factual")
    plt.xlabel("y_factual")
    plt.ylabel("Frequency")
    plt.show()

def compute_propensity_scores(df, confounders):
    """
    Trains a logistic regression model to estimate propensity scores
    based on the specified confounders.
    """
    X = df[confounders]
    y = df['treatment']
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(X, y)
    df['propensity_score'] = ps_model.predict_proba(X)[:, 1]
    return df

def plot_propensity_distribution(df):
    """
    Plots the propensity score distribution by treatment group.
    Returns the matplotlib figure.
    """
    fig = plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='propensity_score', hue='treatment', kde=True, bins=30)
    plt.title("Propensity Score Distribution by Treatment Group")
    return fig

def nearest_neighbor_matching(df, caliper=None):
    """
    Performs 1-to-1 nearest-neighbor matching with strict caliper enforcement.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing the treatment, propensity scores, and covariates.
    caliper : float or None
        Maximum allowed absolute difference in propensity scores for a valid match.
        If None, no caliper is applied.
    
    Returns:
    --------
    matched_df : pd.DataFrame
        Dataframe of matched treated and control units.
    unmatched_treated : int
        Number of treated units that could not be matched.
    """
    # Separate treated and untreated groups
    treated = df[df['treatment'] == 1].copy()
    untreated = df[df['treatment'] == 0].copy()
    
    matched_treated = []
    matched_controls = []
    unmatched_count = 0
    
    # Loop over each treated unit
    for idx, row in treated.iterrows():
        ps_value = row['propensity_score']
        # Calculate absolute differences
        untreated['diff'] = (untreated['propensity_score'] - ps_value).abs()
        # Filter controls within the caliper, if provided
        if caliper is not None:
            valid_controls = untreated[untreated['diff'] <= caliper]
        else:
            valid_controls = untreated
        
        # If no valid control exists within the caliper, skip this treated unit
        if valid_controls.empty:
            unmatched_count += 1
            continue
        
        # Select the control with the smallest difference
        closest_idx = valid_controls['diff'].idxmin()
        matched_treated.append(row)
        matched_controls.append(untreated.loc[closest_idx])
        # Remove the matched control so it won't be reused
        untreated = untreated.drop(closest_idx)
    
    # Combine matched treated and control units
    if matched_treated and matched_controls:
        df_treated = pd.DataFrame(matched_treated)
        df_controls = pd.DataFrame(matched_controls)
        matched_df = pd.concat([df_treated, df_controls])
    else:
        matched_df = pd.DataFrame()  # Return empty DataFrame if no matches
    
    return matched_df, unmatched_count

def estimate_ate(matched_df):
    """
    Estimates the Average Treatment Effect (ATE) on the matched dataset.
    """
    treated_outcome = matched_df[matched_df['treatment'] == 1]['y_factual'].mean()
    control_outcome = matched_df[matched_df['treatment'] == 0]['y_factual'].mean()
    return treated_outcome - control_outcome

def estimate_ate_with_ci(matched_df, alpha=0.05):
    """
    Estimates ATE with confidence intervals.
    """
    treated = matched_df[matched_df['treatment'] == 1]['y_factual']
    control = matched_df[matched_df['treatment'] == 0]['y_factual']
    
    ate = treated.mean() - control.mean()
    se = np.sqrt(treated.var()/len(treated) + control.var()/len(control))
    ci_lower = ate - stats.norm.ppf(1-alpha/2) * se
    ci_upper = ate + stats.norm.ppf(1-alpha/2) * se
    return ate, (ci_lower, ci_upper)

def plot_covariate_balance(matched_df, covariates):
    """
    Plots boxplots of specified covariates by treatment status after matching.
    """
    for covariate in covariates:
        plt.figure()
        sns.boxplot(data=matched_df, x='treatment', y=covariate)
        plt.title(f'{covariate} Distribution by Treatment After Matching')
        plt.show()

def estimate_cate(matched_df, covariate):
    """
    Estimates Conditional ATE (CATE) for a subgroup defined by whether the
    covariate is above or below its median.
    """
    median_val = matched_df[covariate].median()
    matched_df['subgroup'] = (matched_df[covariate] >= median_val).astype(int)
    
    treated_high = matched_df[(matched_df['treatment'] == 1) & (matched_df['subgroup'] == 1)]['y_factual'].mean()
    control_high = matched_df[(matched_df['treatment'] == 0) & (matched_df['subgroup'] == 1)]['y_factual'].mean()
    cate_high = treated_high - control_high
    
    treated_low = matched_df[(matched_df['treatment'] == 1) & (matched_df['subgroup'] == 0)]['y_factual'].mean()
    control_low = matched_df[(matched_df['treatment'] == 0) & (matched_df['subgroup'] == 0)]['y_factual'].mean()
    cate_low = treated_low - control_low
    
    return cate_high, cate_low

def true_effect(df):
    """
    Computes the average true effect using mu1 and mu0 columns.
    """
    return (df['mu1'] - df['mu0']).mean()

def analyze_baseline_characteristics(df):
    """
    Analyzes baseline characteristics between treatment groups.
    """
    import warnings
    summary = pd.DataFrame()
    p_values = {}
    covariates = [f'x{i}' for i in range(1, 26)]
    
    for col in covariates:
        treated = df[df['treatment'] == 1][col]
        control = df[df['treatment'] == 0][col]
        col_summary = pd.DataFrame({
            'Treated_Mean': [treated.mean()],
            'Treated_SD': [treated.std()],
            'Control_Mean': [control.mean()],
            'Control_SD': [control.std()]
        }, index=[col])
        summary = pd.concat([summary, col_summary])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, p_val = stats.ttest_ind(treated, control)
            p_values[col] = p_val
    
    summary['Std_Difference'] = (summary['Treated_Mean'] - summary['Control_Mean']) / \
                                 np.sqrt((summary['Treated_SD']**2 + summary['Control_SD']**2) / 2)
    
    return summary, p_values

def inverse_probability_weighting(df):
    """
    Implements inverse probability weighting for treatment effect estimation.
    """
    weights = np.where(df['treatment'] == 1,
                      1 / df['propensity_score'],
                      1 / (1 - df['propensity_score']))
    percentile_99 = np.percentile(weights, 99)
    weights = np.minimum(weights, percentile_99)
    weights = weights / np.mean(weights)
    
    weighted_treated = np.sum(weights * df['y_factual'] * df['treatment']) / np.sum(weights * df['treatment'])
    weighted_control = np.sum(weights * df['y_factual'] * (1 - df['treatment'])) / np.sum(weights * (1 - df['treatment']))
    ate = weighted_treated - weighted_control
    return ate, weights

def simulate_treatment_effect(df, effect_size):
    """
    Simulates treatment effects by modifying outcomes.
    """
    df_sim = df.copy()
    df_sim.loc[df_sim['treatment'] == 1, 'y_factual'] += effect_size
    return df_sim

def add_validation_metrics(true_ate, estimated_ate, ci):
    """
    Display validation metrics.
    """
    st.subheader("Validation Metrics")
    st.write(f"Bias: {estimated_ate - true_ate:.3f}")
    st.write(f"Coverage: {ci[0] <= true_ate <= ci[1]}")

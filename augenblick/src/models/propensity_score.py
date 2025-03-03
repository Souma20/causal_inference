import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

class PropensityScoreModel:
    """
    Propensity score matching model for causal inference.
    
    This model estimates the propensity scores (probability of receiving treatment)
    and uses these scores to match treated units with similar control units
    to estimate the causal effect.
    """
    
    def __init__(
        self,
        method: str = 'logistic',
        caliper: Optional[float] = 0.2,
        n_neighbors: int = 1,
        random_state: int = 42
    ):
        """
        Initialize the propensity score model.
        
        Args:
            method: Method to estimate propensity scores ('logistic' or 'random_forest')
            caliper: Maximum allowed distance between matched units (in std of propensity)
                    If None, no caliper is applied
            n_neighbors: Number of control units to match with each treated unit
            random_state: Random seed for reproducibility
        """
        self.method = method
        self.caliper = caliper
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.propensity_model = None
        self.propensity_scores = None
        self.treatment_effect = None
        self.matched_data = None
        
        # Initialize propensity model
        if method == 'logistic':
            self.propensity_model = LogisticRegression(random_state=random_state)
        elif method == 'random_forest':
            self.propensity_model = RandomForestClassifier(random_state=random_state)
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'logistic' or 'random_forest'")
    
    def fit(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        confounders: List[str]
    ) -> 'PropensityScoreModel':
        """
        Fit the propensity score model.
        
        Args:
            data: Input DataFrame
            treatment_col: Name of the treatment column
            confounders: List of confounding variable columns
            
        Returns:
            Self, for method chaining
        """
        # Store variables for later use
        self.treatment_col = treatment_col
        self.confounders = confounders
        
        # Fit propensity model
        X = data[confounders]
        y = data[treatment_col]
        
        self.propensity_model.fit(X, y)
        
        # Calculate propensity scores
        self.propensity_scores = self.propensity_model.predict_proba(X)[:, 1]
        
        # Add propensity scores to data
        self.data_with_ps = data.copy()
        self.data_with_ps['propensity_score'] = self.propensity_scores
        
        return self
    
    def match(self) -> pd.DataFrame:
        """
        Perform propensity score matching.
        
        Returns:
            DataFrame with matched samples
        """
        if self.propensity_scores is None:
            raise ValueError("Model must be fitted before matching")
        
        # Extract treatment and control groups
        treated = self.data_with_ps[self.data_with_ps[self.treatment_col] == 1].copy()
        control = self.data_with_ps[self.data_with_ps[self.treatment_col] == 0].copy()
        
        # Calculate caliper width if caliper is specified
        caliper_width = None
        if self.caliper is not None:
            caliper_width = self.caliper * self.propensity_scores.std()
        
        # Perform matching
        matched_indices = []
        
        for idx, treated_unit in treated.iterrows():
            ps_treated = treated_unit['propensity_score']
            
            # Calculate distance in propensity scores
            control['distance'] = abs(control['propensity_score'] - ps_treated)
            
            # Apply caliper if specified
            valid_matches = control
            if caliper_width is not None:
                valid_matches = control[control['distance'] <= caliper_width]
                
                # Skip if no valid matches within caliper
                if len(valid_matches) == 0:
                    continue
            
            # Find n_neighbors closest matches
            matches = valid_matches.nsmallest(self.n_neighbors, 'distance')
            
            # Record matches
            for match_idx, _ in matches.iterrows():
                matched_indices.append((idx, match_idx))
        
        # Create matched dataset
        if not matched_indices:
            raise ValueError("No matches found. Try increasing caliper or n_neighbors")
        
        treated_indices = [t for t, c in matched_indices]
        control_indices = [c for t, c in matched_indices]
        
        matched_treated = self.data_with_ps.loc[treated_indices].copy()
        matched_control = self.data_with_ps.loc[control_indices].copy()
        
        # Store for later use
        self.matched_data = pd.concat([matched_treated, matched_control])
        
        return self.matched_data
    
    def estimate_effect(
        self,
        outcome_col: str,
        method: str = 'mean_difference'
    ) -> Dict[str, float]:
        """
        Estimate the causal effect after matching.
        
        Args:
            outcome_col: Name of the outcome column
            method: Method to estimate effect ('mean_difference' or 'regression')
            
        Returns:
            Dictionary with treatment effect estimates
        """
        if self.matched_data is None:
            self.match()
        
        # Store outcome column
        self.outcome_col = outcome_col
        
        # Extract treated and control outcomes
        treated_outcomes = self.matched_data[self.matched_data[self.treatment_col] == 1][outcome_col]
        control_outcomes = self.matched_data[self.matched_data[self.treatment_col] == 0][outcome_col]
        
        # Estimate treatment effect
        if method == 'mean_difference':
            ate = treated_outcomes.mean() - control_outcomes.mean()
            
            # Calculate standard error and confidence interval
            n_treated = len(treated_outcomes)
            n_control = len(control_outcomes)
            
            pooled_std = np.sqrt(
                ((n_treated - 1) * treated_outcomes.var() + (n_control - 1) * control_outcomes.var()) /
                (n_treated + n_control - 2)
            )
            
            se = pooled_std * np.sqrt(1/n_treated + 1/n_control)
            ci_lower = ate - 1.96 * se
            ci_upper = ate + 1.96 * se
            
            self.treatment_effect = {
                'ate': ate,
                'se': se,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': 2 * (1 - abs(ate / se)) if se > 0 else np.nan
            }
        
        else:
            raise ValueError(f"Unknown estimation method: {method}")
        
        return self.treatment_effect
    
    def plot_propensity_scores(self) -> plt.Figure:
        """
        Plot the distribution of propensity scores.
        
        Returns:
            Matplotlib figure
        """
        if self.propensity_scores is None:
            raise ValueError("Model must be fitted before plotting")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract treatment and control groups
        treated = self.data_with_ps[self.data_with_ps[self.treatment_col] == 1]
        control = self.data_with_ps[self.data_with_ps[self.treatment_col] == 0]
        
        # Plot propensity score distributions
        sns.histplot(
            treated['propensity_score'], 
            color='blue', 
            alpha=0.5, 
            label='Treated',
            ax=ax
        )
        sns.histplot(
            control['propensity_score'], 
            color='red', 
            alpha=0.5, 
            label='Control',
            ax=ax
        )
        
        plt.title('Propensity Score Distribution')
        plt.xlabel('Propensity Score')
        plt.ylabel('Count')
        plt.legend()
        
        return fig
    
    def plot_balance(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot the balance of confounders before and after matching.
        
        Args:
            figsize: Size of the figure
            
        Returns:
            Matplotlib figure
        """
        if self.matched_data is None:
            raise ValueError("Matching must be performed before balance checking")
        
        # Calculate standardized mean differences before matching
        treated_before = self.data_with_ps[self.data_with_ps[self.treatment_col] == 1]
        control_before = self.data_with_ps[self.data_with_ps[self.treatment_col] == 0]
        
        # Calculate standardized mean differences after matching
        treated_after = self.matched_data[self.matched_data[self.treatment_col] == 1]
        control_after = self.matched_data[self.matched_data[self.treatment_col] == 0]
        
        # Calculate standardized differences
        std_diff_before = []
        std_diff_after = []
        
        for col in self.confounders:
            # Before matching
            t_mean = treated_before[col].mean()
            c_mean = control_before[col].mean()
            t_std = treated_before[col].std()
            c_std = control_before[col].std()
            
            pooled_std = np.sqrt((t_std**2 + c_std**2) / 2)
            std_diff = (t_mean - c_mean) / pooled_std if pooled_std != 0 else 0
            std_diff_before.append((col, std_diff))
            
            # After matching
            t_mean = treated_after[col].mean()
            c_mean = control_after[col].mean()
            t_std = treated_after[col].std()
            c_std = control_after[col].std()
            
            pooled_std = np.sqrt((t_std**2 + c_std**2) / 2)
            std_diff = (t_mean - c_mean) / pooled_std if pooled_std != 0 else 0
            std_diff_after.append((col, std_diff))
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Convert to DataFrames for plotting
        df_before = pd.DataFrame(std_diff_before, columns=['variable', 'std_diff'])
        df_after = pd.DataFrame(std_diff_after, columns=['variable', 'std_diff'])
        
        # Plot
        plt.hlines(y=df_before['variable'], xmin=0, xmax=df_before['std_diff'], color='red', alpha=0.5, linewidth=2, label='Before Matching')
        plt.hlines(y=df_after['variable'], xmin=0, xmax=df_after['std_diff'], color='blue', alpha=0.5, linewidth=2, label='After Matching')
        
        # Add reference line at 0
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add labels
        plt.xlabel('Standardized Mean Difference')
        plt.ylabel('Variables')
        plt.title('Covariate Balance Before and After Matching')
        plt.legend()
        
        return fig

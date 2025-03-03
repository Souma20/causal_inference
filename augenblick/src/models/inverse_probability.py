import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

class InverseProbabilityWeightingModel:
    """
    Inverse Probability Weighting (IPW) model for causal inference.
    
    This model estimates the propensity scores (probability of receiving treatment)
    and uses the inverse of these probabilities as weights to create a pseudo-population
    where treatment assignment is independent of confounders.
    """
    
    def __init__(
        self,
        method: str = 'logistic',
        stabilized: bool = True,
        trim_threshold: Optional[float] = 0.01,
        random_state: int = 42
    ):
        """
        Initialize the IPW model.
        
        Args:
            method: Method to estimate propensity scores ('logistic')
            stabilized: Whether to use stabilized weights
            trim_threshold: Threshold for trimming extreme propensity scores
                           (e.g., 0.01 means propensity scores < 0.01 or > 0.99 will be trimmed)
                           If None, no trimming is applied
            random_state: Random seed for reproducibility
        """
        self.method = method
        self.stabilized = stabilized
        self.trim_threshold = trim_threshold
        self.random_state = random_state
        self.propensity_model = None
        self.propensity_scores = None
        self.weights = None
        self.treatment_effect = None
        
        # Initialize propensity model
        if method == 'logistic':
            self.propensity_model = LogisticRegression(random_state=random_state)
        else:
            raise ValueError(f"Unknown method: {method}. Currently only 'logistic' is supported.")
    
    def fit(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        confounders: List[str]
    ) -> 'InverseProbabilityWeightingModel':
        """
        Fit the IPW model.
        
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
        self.data = data.copy()
        
        # Fit propensity model
        X = data[confounders]
        y = data[treatment_col]
        
        self.propensity_model.fit(X, y)
        
        # Calculate propensity scores
        self.propensity_scores = self.propensity_model.predict_proba(X)[:, 1]
        
        # Trim propensity scores if specified
        if self.trim_threshold is not None:
            self.propensity_scores = np.clip(
                self.propensity_scores,
                self.trim_threshold,
                1 - self.trim_threshold
            )
        
        # Add propensity scores to data
        self.data['propensity_score'] = self.propensity_scores
        
        # Calculate treatment probability
        self.treatment_prob = data[treatment_col].mean()
        
        # Calculate weights
        self._calculate_weights()
        
        return self
    
    def _calculate_weights(self) -> None:
        """
        Calculate inverse probability weights.
        """
        # Initialize weights column
        self.data['ipw'] = np.nan
        
        # Calculate weights for treatment and control groups
        treat_idx = self.data[self.treatment_col] == 1
        control_idx = self.data[self.treatment_col] == 0
        
        # Basic IPW weights
        self.data.loc[treat_idx, 'ipw'] = 1 / self.data.loc[treat_idx, 'propensity_score']
        self.data.loc[control_idx, 'ipw'] = 1 / (1 - self.data.loc[control_idx, 'propensity_score'])
        
        # Stabilize weights if specified
        if self.stabilized:
            self.data.loc[treat_idx, 'ipw'] *= self.treatment_prob
            self.data.loc[control_idx, 'ipw'] *= (1 - self.treatment_prob)
        
        # Store weights
        self.weights = self.data['ipw'].values
    
    def estimate_effect(
        self,
        outcome_col: str,
        method: str = 'weighted_mean'
    ) -> Dict[str, float]:
        """
        Estimate the causal effect using IPW.
        
        Args:
            outcome_col: Name of the outcome column
            method: Method to estimate effect ('weighted_mean')
            
        Returns:
            Dictionary with treatment effect estimates
        """
        # Store outcome column
        self.outcome_col = outcome_col
        
        # Estimate treatment effect
        if method == 'weighted_mean':
            # Calculate weighted outcomes
            weighted_treated = self.data[self.data[self.treatment_col] == 1].copy()
            weighted_control = self.data[self.data[self.treatment_col] == 0].copy()
            
            # Calculate weighted means
            weighted_treated_mean = np.average(
                weighted_treated[outcome_col],
                weights=weighted_treated['ipw']
            )
            weighted_control_mean = np.average(
                weighted_control[outcome_col],
                weights=weighted_control['ipw']
            )
            
            ate = weighted_treated_mean - weighted_control_mean
            
            # Calculate bootstrap confidence interval
            bootstrap_ates = []
            n_bootstrap = 500
            n_samples = len(self.data)
            
            for _ in range(n_bootstrap):
                # Resample with replacement
                bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
                bootstrap_data = self.data.iloc[bootstrap_indices]
                
                # Calculate weighted means for bootstrap sample
                bs_treated = bootstrap_data[bootstrap_data[self.treatment_col] == 1]
                bs_control = bootstrap_data[bootstrap_data[self.treatment_col] == 0]
                
                if len(bs_treated) > 0 and len(bs_control) > 0:
                    bs_treated_mean = np.average(
                        bs_treated[outcome_col],
                        weights=bs_treated['ipw']
                    )
                    bs_control_mean = np.average(
                        bs_control[outcome_col],
                        weights=bs_control['ipw']
                    )
                    
                    bootstrap_ates.append(bs_treated_mean - bs_control_mean)
            
            # Calculate statistics from bootstrap
            bootstrap_ates = np.array(bootstrap_ates)
            se = bootstrap_ates.std()
            ci_lower = np.percentile(bootstrap_ates, 2.5)
            ci_upper = np.percentile(bootstrap_ates, 97.5)
            
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
    
    def plot_weights(self) -> plt.Figure:
        """
        Plot the distribution of weights.
        
        Returns:
            Matplotlib figure
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before plotting weights")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract treatment and control groups
        treated = self.data[self.data[self.treatment_col] == 1]
        control = self.data[self.data[self.treatment_col] == 0]
        
        # Plot weight distributions
        sns.histplot(
            treated['ipw'], 
            color='blue', 
            alpha=0.5, 
            label='Treated',
            ax=ax
        )
        sns.histplot(
            control['ipw'], 
            color='red', 
            alpha=0.5, 
            label='Control',
            ax=ax
        )
        
        plt.title('Inverse Probability Weight Distribution')
        plt.xlabel('Weight')
        plt.ylabel('Count')
        plt.legend()
        
        return fig
    
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
        treated = self.data[self.data[self.treatment_col] == 1]
        control = self.data[self.data[self.treatment_col] == 0]
        
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
        
    def balance_check(self) -> pd.DataFrame:
        """
        Check balance of confounders after weighting.
        
        Returns:
            DataFrame with balance statistics
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before balance checking")
        
        balance_stats = []
        
        # Calculate weighted statistics for treatment and control groups
        for col in self.confounders:
            # Calculate unweighted statistics
            treated = self.data[self.data[self.treatment_col] == 1]
            control = self.data[self.data[self.treatment_col] == 0]
            
            t_mean = treated[col].mean()
            c_mean = control[col].mean()
            t_std = treated[col].std()
            c_std = control[col].std()
            
            # Calculate standardized mean difference (unweighted)
            std_pooled = np.sqrt((t_std**2 + c_std**2) / 2)
            std_mean_diff_before = (t_mean - c_mean) / std_pooled if std_pooled != 0 else np.nan
            
            # Calculate weighted statistics
            t_mean_weighted = np.average(treated[col], weights=treated['ipw'])
            c_mean_weighted = np.average(control[col], weights=control['ipw'])
            
            # Calculate weighted variance
            t_var_weighted = np.average((treated[col] - t_mean_weighted)**2, weights=treated['ipw'])
            c_var_weighted = np.average((control[col] - c_mean_weighted)**2, weights=control['ipw'])
            
            t_std_weighted = np.sqrt(t_var_weighted)
            c_std_weighted = np.sqrt(c_var_weighted)
            
            # Calculate standardized mean difference (weighted)
            std_pooled_weighted = np.sqrt((t_std_weighted**2 + c_std_weighted**2) / 2)
            std_mean_diff_after = (t_mean_weighted - c_mean_weighted) / std_pooled_weighted if std_pooled_weighted != 0 else np.nan
            
            balance_stats.append({
                'variable': col,
                'std_mean_diff_before': std_mean_diff_before,
                'std_mean_diff_after': std_mean_diff_after,
                'improvement': abs(std_mean_diff_before) - abs(std_mean_diff_after)
            })
        
        balance_df = pd.DataFrame(balance_stats)
        
        # Sort by improvement
        return balance_df.sort_values('improvement', ascending=False)

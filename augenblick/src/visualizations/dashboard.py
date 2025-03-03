import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any

# Import models and visualization functions
import sys
import os

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.data_loader import load_dataset, generate_synthetic_data
from data_processing.preprocessing import preprocess_data, identify_confounders, balance_check
from models.propensity_score import PropensityScoreModel
from models.inverse_probability import InverseProbabilityWeightingModel
from visualizations.plots import (
    plot_outcome_by_treatment,
    plot_confounder_distribution,
    plot_treatment_effect,
    plot_covariate_balance,
    plot_interactive_balance,
    plot_interactive_propensity
)

def main():
    """
    Main function for the Streamlit dashboard.
    """
    # Set page title and configuration
    st.set_page_config(
        page_title="Causal Inference Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["Home", "Data Upload", "Data Exploration", "Causal Analysis", "Results"]
    )
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'preprocessed_data' not in st.session_state:
        st.session_state.preprocessed_data = None
    if 'model_results' not in st.session_state:
        st.session_state.model_results = {}
    
    # Display selected page
    if page == "Home":
        display_home_page()
    elif page == "Data Upload":
        display_data_upload_page()
    elif page == "Data Exploration":
        display_data_exploration_page()
    elif page == "Causal Analysis":
        display_causal_analysis_page()
    elif page == "Results":
        display_results_page()

def display_home_page():
    """
    Display the home page with introduction to causal inference.
    """
    st.title("Causal Inference in Healthcare")
    
    st.markdown("""
    ## Welcome to the Causal Inference Dashboard
    
    This application helps healthcare researchers and practitioners:
    
    1. **Upload and preprocess healthcare data**
    2. **Explore relationships between treatments and outcomes**
    3. **Apply causal inference methods to estimate treatment effects**
    4. **Visualize and interpret results**
    
    ### Why Causal Inference?
    
    Traditional predictive models only tell us about *correlations* (e.g., "patients who take this medicine tend to recover faster"). 
    However, **causal inference** helps us answer counterfactual questions:
    
    - "Would this patient have recovered if they hadn't taken the medicine?"
    - "Which treatment is truly effective, independent of other factors?"
    
    ### Getting Started
    
    Use the navigation panel on the left to:
    1. Upload your data or generate synthetic data
    2. Explore the data and visualize relationships
    3. Run causal inference analysis
    4. Interpret the results
    
    Let's begin by navigating to the **Data Upload** page.
    """)
    
    # Add a simple diagram to explain causal inference
    st.image("https://images.squarespace-cdn.com/content/v1/60089565df4cf50cbbc19f17/1615040389159-4TL7CG9XFWPRX0YJ3BMA/causal_graph.png", 
             caption="Example of a causal graph showing treatment, outcome, and confounders")

def display_data_upload_page():
    """
    Display the data upload page.
    """
    st.title("Data Upload")
    
    st.markdown("""
    ## Upload Your Data or Generate Synthetic Data
    
    Choose one of the options below to get started:
    """)
    
    # Tabs for different data sources
    tab1, tab2 = st.tabs(["Upload Data", "Generate Synthetic Data"])
    
    with tab1:
        st.markdown("""
        ### Upload Your Dataset
        
        Upload a CSV file containing your healthcare data. The file should include:
        - A treatment variable (binary: 0/1)
        - An outcome variable
        - Potential confounding variables
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded data with {data.shape[0]} rows and {data.shape[1]} columns.")
                st.session_state.data = data
                
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(data.head())
                
                # Show data statistics
                st.subheader("Data Summary")
                st.write(data.describe())
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    with tab2:
        st.markdown("""
        ### Generate Synthetic Data
        
        Create a synthetic dataset for causal inference analysis. You can customize:
        - Number of samples
        - Effect size (true causal effect)
        - Confounder distributions
        """)
        
        # Options for synthetic data generation
        n_samples = st.slider("Number of samples", min_value=100, max_value=10000, value=1000, step=100)
        true_effect = st.slider("True treatment effect", min_value=-5.0, max_value=5.0, value=2.0, step=0.1)
        binary_treatment = st.checkbox("Binary treatment (vs. continuous)", value=True)
        
        # Advanced options
        with st.expander("Advanced options"):
            random_seed = st.number_input("Random seed", min_value=1, max_value=1000, value=42)
            
            # Confounder options
            st.subheader("Confounder settings")
            
            st.markdown("**Age**")
            age_mean = st.slider("Mean age", min_value=20, max_value=80, value=50)
            age_std = st.slider("Age standard deviation", min_value=1, max_value=20, value=10)
            
            st.markdown("**Blood Pressure**")
            bp_mean = st.slider("Mean BP", min_value=80, max_value=160, value=120)
            bp_std = st.slider("BP standard deviation", min_value=1, max_value=20, value=10)
            
            st.markdown("**BMI**")
            bmi_mean = st.slider("Mean BMI", min_value=15, max_value=40, value=25)
            bmi_std = st.slider("BMI standard deviation", min_value=1, max_value=10, value=3)
        
        # Generate button
        if st.button("Generate Data"):
            confounders = {
                'age': {'mean': age_mean, 'std': age_std},
                'blood_pressure': {'mean': bp_mean, 'std': bp_std},
                'bmi': {'mean': bmi_mean, 'std': bmi_std}
            }
            
            data = generate_synthetic_data(
                n_samples=n_samples,
                binary_treatment=binary_treatment,
                confounders=confounders,
                treatment_effect=true_effect,
                random_state=random_seed
            )
            
            st.session_state.data = data
            
            # Show data preview
            st.subheader("Generated Data Preview")
            st.dataframe(data.head())
            
            # Show data statistics
            st.subheader("Data Summary")
            st.write(data.describe())
            
            st.success(f"Successfully generated synthetic data with {data.shape[0]} rows.")
            
            # Display the true effect for reference
            st.info(f"The true causal effect in this synthetic data is: {true_effect}")
    
    # Navigation buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Home"):
            st.session_state.page = "Home"
            st.experimental_rerun()
    with col2:
        if st.session_state.data is not None:
            if st.button("Proceed to Data Exploration →"):
                st.session_state.page = "Data Exploration"
                st.experimental_rerun()

def display_data_exploration_page():
    """
    Display the data exploration page.
    """
    st.title("Data Exploration")
    
    if st.session_state.data is None:
        st.warning("No data loaded. Please upload or generate data first.")
        if st.button("Go to Data Upload"):
            st.session_state.page = "Data Upload"
            st.experimental_rerun()
        return
    
    data = st.session_state.data
    
    st.markdown("""
    ## Explore Your Data
    
    Exploring your data before causal analysis helps you understand:
    - The distribution of treatment and outcome variables
    - Potential confounding variables
    - Relationships between variables
    
    ### Step 1: Select Variables
    
    First, identify the treatment, outcome, and potential confounding variables in your dataset.
    """)
    
    # Variable selection
    col1, col2 = st.columns(2)
    with col1:
        treatment_col = st.selectbox("Select treatment variable", data.columns, 
                                     index=data.columns.get_loc("treatment") if "treatment" in data.columns else 0)
    with col2:
        outcome_col = st.selectbox("Select outcome variable", data.columns,
                                  index=data.columns.get_loc("outcome") if "outcome" in data.columns else 0)
    
    # Multi-select for confounders
    potential_confounders = [col for col in data.columns if col not in [treatment_col, outcome_col]]
    selected_confounders = st.multiselect(
        "Select potential confounding variables",
        potential_confounders,
        default=potential_confounders if len(potential_confounders) <= 5 else potential_confounders[:5]
    )
    
    if not selected_confounders:
        st.warning("Please select at least one confounder to proceed.")
        return
    
    # Store selections in session state
    st.session_state.treatment_col = treatment_col
    st.session_state.outcome_col = outcome_col
    st.session_state.selected_confounders = selected_confounders
    
    # Exploratory visualizations
    st.markdown("### Basic Statistics")
    
    # Treatment distribution
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Treatment Distribution ({treatment_col})")
        treatment_counts = data[treatment_col].value_counts().reset_index()
        treatment_counts.columns = [treatment_col, 'Count']
        st.bar_chart(treatment_counts.set_index(treatment_col))
    
    # Outcome distribution
    with col2:
        st.subheader(f"Outcome Distribution ({outcome_col})")
        if data[outcome_col].nunique() <= 10:  # Categorical outcome
            outcome_counts = data[outcome_col].value_counts().reset_index()
            outcome_counts.columns = [outcome_col, 'Count']
            st.bar_chart(outcome_counts.set_index(outcome_col))
        else:  # Continuous outcome
            fig, ax = plt.subplots()
            data[outcome_col].hist(bins=20, ax=ax)
            st.pyplot(fig)
    
    # Outcome by treatment
    st.subheader("Outcome by Treatment")
    outcome_treatment_fig = plot_outcome_by_treatment(data, treatment_col, outcome_col)
    st.pyplot(outcome_treatment_fig)
    
    # Confounder distributions
    st.subheader("Confounder Distributions by Treatment")
    confounder_dist_fig = plot_confounder_distribution(data, treatment_col, selected_confounders)
    st.pyplot(confounder_dist_fig)
    
    # Correlation heatmap
    st.subheader("Correlation Matrix")
    selected_cols = [treatment_col, outcome_col] + selected_confounders
    corr_matrix = data[selected_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Potential confounders identification
    st.subheader("Potential Confounders Identification")
    
    if st.button("Identify Confounders"):
        auto_confounders = identify_confounders(data, treatment_col, outcome_col)
        
        if auto_confounders:
            st.success(f"Identified potential confounders: {', '.join(auto_confounders)}")
            
            # Showing correlation with treatment and outcome
            conf_corr = pd.DataFrame({
                'variable': auto_confounders,
                'correlation_with_treatment': [data[conf].corr(data[treatment_col]) for conf in auto_confounders],
                'correlation_with_outcome': [data[conf].corr(data[outcome_col]) for conf in auto_confounders]
            })
            
            st.dataframe(conf_corr)
            
            # Option to use these confounders
            if st.button("Use Identified Confounders"):
                st.session_state.selected_confounders = auto_confounders
                st.experimental_rerun()
        else:
            st.warning("No strong confounders identified based on correlation analysis.")
    
    # Navigation buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Data Upload"):
            st.session_state.page = "Data Upload"
            st.experimental_rerun()
    with col2:
        if st.button("Proceed to Causal Analysis →"):
            st.session_state.page = "Causal Analysis"
            st.experimental_rerun()

def display_causal_analysis_page():
    """
    Display the causal analysis page.
    """
    st.title("Causal Analysis")
    
    if st.session_state.data is None:
        st.warning("No data loaded. Please upload or generate data first.")
        if st.button("Go to Data Upload"):
            st.session_state.page = "Data Upload"
            st.experimental_rerun()
        return
    
    if not hasattr(st.session_state, 'treatment_col') or not hasattr(st.session_state, 'outcome_col'):
        st.warning("Variables not selected. Please complete the data exploration step first.")
        if st.button("Go to Data Exploration"):
            st.session_state.page = "Data Exploration"
            st.experimental_rerun()
        return
    
    data = st.session_state.data
    treatment_col = st.session_state.treatment_col
    outcome_col = st.session_state.outcome_col
    selected_confounders = st.session_state.selected_confounders
    
    st.markdown("""
    ## Causal Inference Analysis
    
    Now we'll apply causal inference methods to estimate the treatment effect:
    
    1. **Preprocess** the data
    2. **Choose and configure** causal inference models
    3. **Run the analysis** to estimate treatment effects
    """)
    
    # Data preprocessing
    st.subheader("Step 1: Data Preprocessing")
    
    # Preprocessing options
    with st.expander("Preprocessing Options", expanded=True):
        scale_numeric = st.checkbox("Standardize numeric features", value=True)
        handle_missing = st.checkbox("Handle missing values", value=True)
        
        # Identify categorical variables
        potential_categorical = [col for col in selected_confounders 
                                if data[col].nunique() < 10 or data[col].dtype == 'object']
        categorical_cols = st.multiselect(
            "Select categorical variables (for encoding)",
            selected_confounders,
            default=potential_categorical
        )
    
    # Preprocess button
    if st.button("Preprocess Data"):
        with st.spinner("Preprocessing data..."):
            try:
                preprocessed_data, preprocessing_info = preprocess_data(
                    data=data,
                    treatment_col=treatment_col,
                    outcome_col=outcome_col,
                    confounder_cols=selected_confounders,
                    categorical_cols=categorical_cols,
                    scale_numeric=scale_numeric,
                    handle_missing=handle_missing
                )
                
                st.session_state.preprocessed_data = preprocessed_data
                st.session_state.preprocessing_info = preprocessing_info
                
                st.success("Data preprocessing completed successfully!")
                
                # Show preprocessed data preview
                st.subheader("Preprocessed Data Preview")
                st.dataframe(preprocessed_data.head())
                
                # Show preprocessing information
                st.subheader("Preprocessing Information")
                preprocessing_steps = []
                
                if 'numeric_imputer' in preprocessing_info:
                    preprocessing_steps.append("✅ Missing numeric values imputed")
                
                if 'categorical_imputer' in preprocessing_info:
                    preprocessing_steps.append("✅ Missing categorical values imputed")
                
                if 'encoder' in preprocessing_info:
                    preprocessing_steps.append(f"✅ Categorical variables encoded: {', '.join(categorical_cols)}")
                
                if 'scaler' in preprocessing_info:
                    preprocessing_steps.append("✅ Numeric features standardized")
                
                for step in preprocessing_steps:
                    st.write(step)
                
            except Exception as e:
                st.error(f"Error during preprocessing: {e}")
    
    # Model selection and configuration
    st.subheader("Step 2: Choose Causal Inference Models")
    
    if st.session_state.preprocessed_data is None:
        st.warning("Please preprocess the data before continuing.")
        return
    
    preprocessed_data = st.session_state.preprocessed_data
    
    # Model selection
    models_to_run = {}
    
    with st.expander("Propensity Score Matching", expanded=True):
        run_psm = st.checkbox("Run Propensity Score Matching analysis", value=True)
        
        if run_psm:
            psm_method = st.selectbox(
                "Propensity score estimation method",
                ["logistic", "random_forest"],
                index=0
            )
            
            psm_caliper = st.slider(
                "Matching caliper (as a fraction of propensity score std)",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05
            )
            
            psm_n_neighbors = st.number_input(
                "Number of neighbors to match",
                min_value=1,
                max_value=5,
                value=1
            )
            
            models_to_run["psm"] = {
                "method": psm_method,
                "caliper": psm_caliper,
                "n_neighbors": psm_n_neighbors
            }
    
    with st.expander("Inverse Probability Weighting", expanded=True):
        run_ipw = st.checkbox("Run Inverse Probability Weighting analysis", value=True)
        
        if run_ipw:
            ipw_stabilized = st.checkbox("Use stabilized weights", value=True)
            
            ipw_trim = st.slider(
                "Trim threshold for extreme propensity scores",
                min_value=0.0,
                max_value=0.2,
                value=0.01,
                step=0.01
            )
            
            models_to_run["ipw"] = {
                "stabilized": ipw_stabilized,
                "trim_threshold": ipw_trim
            }
    
    # Run analysis button
    if models_to_run and st.button("Run Causal Analysis"):
        with st.spinner("Running causal inference analysis..."):
            results = {}
            
            try:
                # Run Propensity Score Matching
                if "psm" in models_to_run:
                    psm_params = models_to_run["psm"]
                    st.write("Running Propensity Score Matching...")
                    
                    psm_model = PropensityScoreModel(
                        method=psm_params["method"],
                        caliper=psm_params["caliper"],
                        n_neighbors=psm_params["n_neighbors"]
                    )
                    
                    # Fit model
                    psm_model.fit(
                        data=preprocessed_data,
                        treatment_col=treatment_col,
                        confounders=selected_confounders
                    )
                    
                    # Perform matching
                    matched_data = psm_model.match()
                    
                    # Estimate effect
                    effect = psm_model.estimate_effect(outcome_col=outcome_col)
                    
                    # Store results
                    results["Propensity Score Matching"] = {
                        "model": psm_model,
                        "effect": effect,
                        "matched_data": matched_data
                    }
                    
                    st.success(f"Propensity Score Matching completed. Estimated ATE: {effect['ate']:.4f} (95% CI: [{effect['ci_lower']:.4f}, {effect['ci_upper']:.4f}])")
                
                # Run Inverse Probability Weighting
                if "ipw" in models_to_run:
                    ipw_params = models_to_run["ipw"]
                    st.write("Running Inverse Probability Weighting...")
                    
                    ipw_model = InverseProbabilityWeightingModel(
                        stabilized=ipw_params["stabilized"],
                        trim_threshold=ipw_params["trim_threshold"]
                    )
                    
                    # Fit model
                    ipw_model.fit(
                        data=preprocessed_data,
                        treatment_col=treatment_col,
                        confounders=selected_confounders
                    )
                    
                    # Estimate effect
                    effect = ipw_model.estimate_effect(outcome_col=outcome_col)
                    
                    # Store results
                    results["Inverse Probability Weighting"] = {
                        "model": ipw_model,
                        "effect": effect
                    }
                    
                    st.success(f"Inverse Probability Weighting completed. Estimated ATE: {effect['ate']:.4f} (95% CI: [{effect['ci_lower']:.4f}, {effect['ci_upper']:.4f}])")
                
                # Store results in session state
                st.session_state.model_results = results
                
                # Display a summary of results
                st.subheader("Analysis Results Summary")
                
                results_df = pd.DataFrame({
                    model_name: {
                        "ATE": f"{model_info['effect']['ate']:.4f}",
                        "SE": f"{model_info['effect']['se']:.4f}",
                        "95% CI": f"[{model_info['effect']['ci_lower']:.4f}, {model_info['effect']['ci_upper']:.4f}]",
                        "p-value": f"{model_info['effect']['p_value']:.4f}" if 'p_value' in model_info['effect'] else "N/A"
                    }
                    for model_name, model_info in results.items()
                }).transpose()
                
                st.dataframe(results_df)
            
            except Exception as e:
                st.error(f"Error during analysis: {e}")
    
    # Navigation buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Data Exploration"):
            st.session_state.page = "Data Exploration"
            st.experimental_rerun()
    with col2:
        if st.session_state.model_results:
            if st.button("Proceed to Results →"):
                st.session_state.page = "Results"
                st.experimental_rerun()

def display_results_page():
    """
    Display the results page.
    """
    st.title("Results and Interpretation")
    
    if not st.session_state.model_results:
        st.warning("No analysis results found. Please run the causal analysis first.")
        if st.button("Go to Causal Analysis"):
            st.session_state.page = "Causal Analysis"
            st.experimental_rerun()
        return
    
    st.markdown("""
    ## Causal Inference Results
    
    This page presents the results of your causal inference analysis:
    
    1. **Treatment Effect Estimates** from different methods
    2. **Balance Diagnostics** to assess the quality of adjustment
    3. **Propensity Score Distributions** to evaluate overlap
    4. **Interpretation** of the findings
    """)
    
    # Treatment effect visualization
    st.subheader("Treatment Effect Estimates")
    
    effect_estimates = {
        model_name: model_info["effect"]
        for model_name, model_info in st.session_state.model_results.items()
    }
    
    effect_fig = plot_treatment_effect(effect_estimates)
    st.pyplot(effect_fig)
    
    # Detailed results for each model
    st.markdown("### Detailed Model Results")
    
    for model_name, model_info in st.session_state.model_results.items():
        with st.expander(f"{model_name} Results", expanded=True):
            # Effect estimates
            st.write("**Treatment Effect Estimates:**")
            effect = model_info["effect"]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Treatment Effect (ATE)", f"{effect['ate']:.4f}")
            col2.metric("Standard Error", f"{effect['se']:.4f}")
            col3.metric("p-value", f"{effect['p_value']:.4f}" if 'p_value' in effect else "N/A")
            
            st.write(f"95% Confidence Interval: [{effect['ci_lower']:.4f}, {effect['ci_upper']:.4f}]")
            
            # Model-specific visualizations
            if model_name == "Propensity Score Matching":
                model = model_info["model"]
                
                # Propensity score distribution
                st.write("**Propensity Score Distribution:**")
                ps_fig = model.plot_propensity_scores()
                st.pyplot(ps_fig)
                
                # Balance plot
                st.write("**Covariate Balance Before and After Matching:**")
                balance_fig = model.plot_balance()
                st.pyplot(balance_fig)
                
                # Matched data
                st.write("**Matched Data Summary:**")
                matched_data = model_info["matched_data"]
                st.write(f"Number of matched pairs: {len(matched_data[matched_data[st.session_state.treatment_col] == 1])}")
                
                # Outcome distribution in matched data
                st.write("**Outcome Distribution in Matched Data:**")
                outcome_matched_fig = plot_outcome_by_treatment(
                    matched_data, 
                    st.session_state.treatment_col, 
                    st.session_state.outcome_col
                )
                st.pyplot(outcome_matched_fig)
            
            elif model_name == "Inverse Probability Weighting":
                model = model_info["model"]
                
                # Propensity score distribution
                st.write("**Propensity Score Distribution:**")
                ps_fig = model.plot_propensity_scores()
                st.pyplot(ps_fig)
                
                # Weight distribution
                st.write("**Weight Distribution:**")
                weights_fig = model.plot_weights()
                st.pyplot(weights_fig)
                
                # Balance check
                st.write("**Covariate Balance Before and After Weighting:**")
                balance_df = model.balance_check()
                
                # Plot balance
                balance_fig = plot_covariate_balance(balance_df)
                st.pyplot(balance_fig)
                
                # Interactive balance plot with plotly
                st.write("**Interactive Balance Plot:**")
                interactive_balance = plot_interactive_balance(balance_df)
                st.plotly_chart(interactive_balance)
    
    # Interpretation guidance
    st.markdown("### Interpretation of Results")
    
    st.markdown("""
    #### Key Findings
    
    The causal effect estimates represent the *average treatment effect (ATE)* - the estimated effect of treatment on the outcome across the population.
    
    **How to interpret the results:**
    
    1. **Point Estimate (ATE)**: The estimated average effect of treatment on the outcome.
    
    2. **Confidence Interval**: The range within which the true effect likely lies (with 95% confidence).
       - If the interval does not contain zero, the effect is statistically significant.
       - The width of the interval indicates precision of the estimate.
    
    3. **Balance Diagnostics**: 
       - Good balance is indicated by standardized mean differences close to zero.
       - Values within ±0.1 (green reference lines) are generally considered acceptable.
    
    4. **Propensity Scores**:
       - Good overlap between treatment and control groups indicates that the model can find appropriate comparisons.
       - Poor overlap may indicate potential issues with the estimates.
    """)
    
    # Limitations
    st.markdown("#### Limitations")
    
    st.markdown("""
    Causal inference from observational data requires careful interpretation:
    
    1. **Unobserved Confounding**: These methods assume all confounders are measured. If important confounders are missing, estimates may be biased.
    
    2. **Positivity Assumption**: All subjects must have a non-zero probability of receiving each treatment level. Areas with poor overlap may violate this assumption.
    
    3. **Model Specification**: Results depend on correct specification of the propensity score model.
    
    4. **Extrapolation**: Be cautious about generalizing results beyond the study population.
    """)
    
    # Export results
    st.subheader("Export Results")
    
    if st.button("Generate Report Summary"):
        # Create report summary
        report = "# Causal Inference Analysis Report\n\n"
        report += "## Treatment Effect Estimates\n\n"
        
        for model_name, model_info in st.session_state.model_results.items():
            effect = model_info["effect"]
            
            report += f"### {model_name}\n"
            report += f"- Average Treatment Effect (ATE): {effect['ate']:.4f}\n"
            report += f"- Standard Error: {effect['se']:.4f}\n"
            report += f"- 95% Confidence Interval: [{effect['ci_lower']:.4f}, {effect['ci_upper']:.4f}]\n"
            report += f"- p-value: {effect['p_value']:.4f if 'p_value' in effect else 'N/A'}\n\n"
        
        # Display report
        st.download_button(
            label="Download Report",
            data=report,
            file_name="causal_inference_report.md",
            mime="text/markdown"
        )
    
    # Navigation button
    st.markdown("---")
    if st.button("← Back to Causal Analysis"):
        st.session_state.page = "Causal Analysis"
        st.experimental_rerun()

if __name__ == "__main__":
    main()

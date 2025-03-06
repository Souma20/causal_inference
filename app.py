#app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from causal_analysis import (
    load_data, 
    compute_propensity_scores, 
    nearest_neighbor_matching, 
    estimate_ate, 
    estimate_ate_with_ci,
    estimate_cate, 
    true_effect,
    inverse_probability_weighting,
    analyze_baseline_characteristics,
    load_ihdp_data,
    simulate_treatment_effect,
    add_validation_metrics,
    plot_propensity_distribution
)

def create_streamlit_app():
    st.title("Causal Inference Analysis Dashboard")
    
    # File upload
    uploaded_file = st.file_uploader("Upload IHDP dataset", type=['csv'])
    if uploaded_file is not None:
        df, descriptions = load_ihdp_data(uploaded_file)
        
        # Compute propensity scores using confounders x1 to x25
        confounders = [f'x{i}' for i in range(1, 26)]
        df_with_ps = compute_propensity_scores(df, confounders)
        
        # Sidebar for analysis options
        st.sidebar.header("Analysis Options")
        analysis_type = st.sidebar.selectbox(
            "Select Analysis Type",
            ["Data Overview", "Propensity Score Analysis", "Treatment Effects", "Subgroup Analysis", "What-if Analysis"]
        )
        
        if analysis_type == "Data Overview":
            st.header("Dataset Overview")
            st.write(f"Dataset shape: {df.shape}")
            st.write("Column descriptions:", descriptions)
            st.subheader("Dataset Summary")
            st.write(df.describe())
            st.subheader("Baseline Characteristics")
            summary, p_values = analyze_baseline_characteristics(df)
            st.write(summary)
            
        elif analysis_type == "Propensity Score Analysis":
            st.header("Propensity Score Analysis")
            fig_ps = plot_propensity_distribution(df_with_ps)
            st.pyplot(fig_ps)
            plt.close()
            
        elif analysis_type == "Treatment Effects":
            st.header("Treatment Effect Estimates")
            st.subheader("Propensity Score Matching")
            
            # Caliper selection slider (allowing strict values)
            caliper = st.slider(
                "Select caliper for matching (0 = no caliper)", 
                min_value=0.0, 
                max_value=0.5, 
                value=0.2, 
                step=0.01,
                key='caliper_slider'
            )
            
            # Get matched data and count unmatched treated units
            matched_df, unmatched_count = nearest_neighbor_matching(df_with_ps, caliper if caliper > 0 else None)
            n_treated_original = len(df_with_ps[df_with_ps['treatment'] == 1])
            n_matched = len(matched_df[matched_df['treatment'] == 1])
            match_rate = (n_matched / n_treated_original) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Original Treated Units", n_treated_original)
            with col2:
                st.metric("Matched Treated Units", n_matched)
            with col3:
                st.metric("Unmatched Treated Units", unmatched_count)
            with col4:
                st.metric("Match Rate", f"{match_rate:.1f}%")
            
            st.subheader("Effect Estimates")
            est_col1, est_col2, est_col3 = st.columns(3)
            
            with est_col1:
                st.markdown("**Matching Estimate**")
                ate_matching = estimate_ate(matched_df)
                ate, ci = estimate_ate_with_ci(matched_df)
                st.metric("ATE", f"{ate_matching:.3f}")
                st.write(f"95% CI: ({ci[0]:.3f}, {ci[1]:.3f})")
            
            with est_col2:
                st.markdown("**IPW Estimate**")
                ate_ipw, weights = inverse_probability_weighting(df_with_ps)
                st.metric("ATE", f"{ate_ipw:.3f}")
                st.write(f"Weight range: [{weights.min():.2f}, {weights.max():.2f}]")
            
            with est_col3:
                st.markdown("**True Effect**")
                true_ate = true_effect(df)
                st.metric("ATE", f"{true_ate:.3f}")
            
            st.subheader("Match Rate vs. Caliper")
            # Plot match rate across a range of caliper values
            caliper_values = np.linspace(0.0, 0.5, 20)
            match_rates = []
            for cal in caliper_values:
                _, unmatched = nearest_neighbor_matching(df_with_ps, cal)
                matched_treated = len(df_with_ps[df_with_ps['treatment'] == 1]) - unmatched
                rate = (matched_treated / n_treated_original) * 100
                match_rates.append(rate)
            
            fig_match, ax_match = plt.subplots(figsize=(8, 4))
            ax_match.plot(caliper_values, match_rates, marker='o')
            ax_match.set_xlabel("Caliper Value")
            ax_match.set_ylabel("Match Rate (%)")
            ax_match.set_title("Match Rate vs. Caliper")
            st.pyplot(fig_match)
            plt.close()
            
            st.subheader("Balance Visualization")
            tab1, tab2 = st.tabs(["Covariate Balance", "Treatment Effect"])
            
            with tab1:
                display_covariates = confounders[:3]
                fig_balance = plt.figure(figsize=(12, 4))
                for i, cov in enumerate(display_covariates):
                    plt.subplot(1, 3, i+1)
                    sns.boxplot(data=matched_df, x='treatment', y=cov)
                    plt.title(f'{cov} Distribution')
                    plt.xlabel('Treatment')
                st.pyplot(fig_balance)
                plt.close()
            
            with tab2:
                fig_effect = plt.figure(figsize=(10, 6))
                sns.boxplot(data=matched_df, x='treatment', y='y_factual')
                plt.title("Outcome Distribution by Treatment Group")
                plt.xlabel("Treatment")
                plt.ylabel("Outcome")
                st.pyplot(fig_effect)
                plt.close()
            
            st.subheader("Standardized Differences")
            summary, _ = analyze_baseline_characteristics(matched_df)
            fig_std_diff = plt.figure(figsize=(10, 6))
            sns.barplot(x=summary.index, y='Std_Difference', data=summary)
            plt.axhline(y=0.1, color='r', linestyle='--', label='Threshold (0.1)')
            plt.axhline(y=-0.1, color='r', linestyle='--')
            plt.xticks(rotation=45)
            plt.title("Standardized Differences After Matching")
            st.pyplot(fig_std_diff)
            plt.close()
            
        elif analysis_type == "Subgroup Analysis":
            st.header("Subgroup Analysis")
            # Perform matching without caliper for subgroup analysis
            matched_df, _ = nearest_neighbor_matching(df_with_ps)
            selected_covariate = st.selectbox("Select covariate for subgroup analysis", options=confounders)
            if selected_covariate:
                cate_high, cate_low = estimate_cate(matched_df, selected_covariate)
                st.write(f"CATE for {selected_covariate} ≥ median: {cate_high:.3f}")
                st.write(f"CATE for {selected_covariate} < median: {cate_low:.3f}")
                fig_subgroup = plt.figure(figsize=(10, 6))
                sns.boxplot(data=matched_df, x='treatment', y='y_factual', hue=selected_covariate)
                plt.title(f"Treatment Effect by {selected_covariate} Subgroups")
                st.pyplot(fig_subgroup)
                plt.close()
                
        elif analysis_type == "What-if Analysis":
            st.header("What-if Analysis")
            # For simulation, use matched data
            matched_df, _ = nearest_neighbor_matching(df_with_ps)
            effect_modifier = st.slider("Modify treatment effect by:", min_value=-5.0, max_value=5.0, value=0.0, step=0.5)
            if effect_modifier != 0:
                simulated_df = simulate_treatment_effect(matched_df, effect_modifier)
                sim_ate, sim_ci = estimate_ate_with_ci(simulated_df)
                st.write(f"Simulated ATE: {sim_ate:.3f}")
                st.write(f"95% CI: ({sim_ci[0]:.3f}, {sim_ci[1]:.3f})")
                true_ate = true_effect(df)
                add_validation_metrics(true_ate, sim_ate, sim_ci)
            
    else:
        st.info("Please upload your CSV file to get started.")

if __name__ == "__main__":
    create_streamlit_app()

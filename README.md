# Causal Inference Project

This project implements a causal inference analysis pipeline for healthcare data, with a focus on estimating treatment effects using propensity score methods and inverse probability weighting (IPW). It includes interactive dashboards and what-if analysis tools to help users explore, visualize, and validate the causal effects of interventions.

# Causal Inference Analysis: IHDP Dataset
## 📊 Project Overview

An interactive causal inference analysis tool that estimates the impact of specialized childcare interventions on infants' cognitive outcomes using the Infant Health and Development Program (IHDP) dataset. This project combines advanced causal inference techniques with an intuitive web interface for exploring and understanding treatment effects.

![Screenshot 2025-03-06 155348](https://github.com/user-attachments/assets/ea8b0307-de3d-4c1b-bba7-f73988fe7969)
![Screenshot 2025-03-06 155359](https://github.com/user-attachments/assets/32f062df-2b34-4399-84c1-eede6846b627)

## 🎯 Key Features

- **Interactive Dashboard** built with Streamlit
- **Multiple Causal Inference Methods**:
  - Propensity Score Matching (PSM)
  - Inverse Probability Weighting (IPW)
- **Real-time What-if Analysis**
- **Dynamic Visualization** of treatment effects
- **Comprehensive Statistical Reports**

## 🔧 Technologies Used

- Python 3.12
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Statsmodels

## 📈 Dataset Information

The IHDP dataset contains:
- 747 samples
- 25 pre-intervention covariates
- Treatment variable (specialized childcare intervention)
- Outcome variable (cognitive test scores at age 5)


## 📊 Features & Screenshots

### 1. Propensity Score Analysis
![Screenshot 2025-03-06 155412](https://github.com/user-attachments/assets/e14b62fc-8814-4468-bc11-dd1988a6bb32)
![Screenshot 2025-03-06 155429](https://github.com/user-attachments/assets/920e6144-a80d-4daf-a3bf-dccf58487403)

### 2. Treatment Effect Estimation
![Screenshot 2025-03-06 161346](https://github.com/user-attachments/assets/192ae9fb-a850-4c1c-ae18-c37bdfd1527f)

### 3. Covariate Balance
![Screenshot 2025-03-06 160653](https://github.com/user-attachments/assets/f0a67ea7-d370-44eb-bc6f-3cb9a074975e)

### 4. What-if Simulation
![Screenshot 2025-03-06 160351](https://github.com/user-attachments/assets/1f71aa87-3d49-4a09-a4d1-0dc674e1f46c)

## 🔍 Methodology

1. **Data Preprocessing**
   - Cleaning and validation
   - Feature engineering
   - Missing value handling

2. **Causal Analysis**
   - Propensity score estimation
   - Matching algorithms
   - Treatment effect calculation
   - Sensitivity analysis

3. **Visualization & Reporting**
   - Interactive plots
   - Statistical summaries
   - Real-time analysis updates

## 📈 Results

![Results Summary](assets/results-summary.png)

Our analysis shows:
- Estimated Average Treatment Effect (ATE)
- Confidence intervals
- Subgroup analyses
- Sensitivity measures


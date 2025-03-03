# Causal Inference in Healthcare

This application implements a causal inference framework for healthcare data analysis, allowing researchers and practitioners to estimate the causal effect of treatments on patient outcomes while controlling for confounding variables.

## 🌟 Features

- **Data Management**: Upload your own healthcare data or generate synthetic data for testing
- **Data Exploration**: Visualize relationships between treatments, outcomes, and potential confounders
- **Causal Analysis**: Apply multiple causal inference methods:
  - Propensity Score Matching
  - Inverse Probability Weighting
- **Results Visualization**: Interactive visualizations of treatment effects and diagnostics
- **Interpretability**: Guidance on interpreting causal estimates and their limitations

## 📋 Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/causal_inference.git
   cd causal_inference
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Running the Dashboard

1. Launch the Streamlit dashboard:
   ```bash
   cd dashboard
   streamlit run app.py
   ```

2. Open your browser and navigate to http://localhost:8501

### Using the Dashboard

The dashboard guides you through:

1. **Data Upload/Generation**: Upload your healthcare data or generate synthetic data
2. **Data Exploration**: Select treatment, outcome, and confounding variables
3. **Causal Analysis**: Configure and run causal inference methods
4. **Results Interpretation**: Explore estimated treatment effects and diagnostic visualizations

### Programmatic Usage

You can also use the causal inference models programmatically:

```python
from src.data_processing.data_loader import generate_synthetic_data
from src.models.propensity_score import PropensityScoreModel

# Generate synthetic data
data = generate_synthetic_data(
    n_samples=1000,
    binary_treatment=True,
    treatment_effect=2.0
)

# Initialize model
psm = PropensityScoreModel(method='logistic', caliper=0.2)

# Fit model
psm.fit(
    data=data,
    treatment_col='treatment',
    confounders=['age', 'blood_pressure', 'bmi']
)

# Perform matching
matched_data = psm.match()

# Estimate effect
effect = psm.estimate_effect(outcome_col='outcome')
print(f"Estimated treatment effect: {effect['ate']}")
```

## 📊 Example

![Dashboard Screenshot](docs/dashboard_screenshot.png)

## 📚 Understanding Causal Inference

Causal inference methods help estimate the causal effect of a treatment on an outcome by:

1. **Addressing confounding**: Controlling for variables that affect both treatment assignment and outcomes
2. **Creating balance**: Making treated and control groups comparable
3. **Estimating counterfactuals**: What would have happened if treated units had not received treatment (and vice versa)

## 🛠️ Project Structure

```
causal_inference/
├── dashboard/           # Streamlit dashboard application
├── data/                # Data storage (example and user-uploaded data)
├── docs/                # Documentation
├── notebooks/           # Jupyter notebooks for exploration
├── src/                 # Source code
│   ├── data_processing/ # Data loading and preprocessing
│   ├── models/          # Causal inference models
│   └── visualizations/  # Visualization utilities
└── tests/               # Unit tests
```

## 📖 Citation

If you use this project in your research, please cite:

```
@software{causal_inference_healthcare,
  author = {Your Name},
  title = {Causal Inference in Healthcare},
  year = {2023},
  url = {https://github.com/yourusername/causal_inference}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 
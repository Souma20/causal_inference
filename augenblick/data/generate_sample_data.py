import os
import sys
import pandas as pd
import numpy as np
import argparse

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Import the data generation function
from data_processing.data_loader import generate_synthetic_data

def save_dataset(data, filename, output_dir='.'):
    """Save a dataset to a file."""
    output_path = os.path.join(output_dir, filename)
    data.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    return output_path

def generate_heart_disease_dataset(n_samples=1000, random_state=42):
    """
    Generate a synthetic dataset mimicking heart disease data.
    
    Features:
    - Age: age in years
    - Sex: 0 = female, 1 = male
    - ChestPain: chest pain type (0-3)
    - RestBP: resting blood pressure
    - Cholesterol: serum cholesterol in mg/dl
    - BloodSugar: fasting blood sugar > 120 mg/dl (0 = false, 1 = true)
    - MaxHR: maximum heart rate achieved
    - Exercise: exercise induced angina (0 = no, 1 = yes)
    - Treatment: if the patient received treatment (0 = no, 1 = yes)
    - Outcome: heart disease status or recovery metric
    """
    np.random.seed(random_state)
    
    # Define confounders (they affect both treatment and outcome)
    confounders = {
        'Age': {'mean': 54, 'std': 9},
        'RestBP': {'mean': 131, 'std': 17},
        'Cholesterol': {'mean': 246, 'std': 51},
        'MaxHR': {'mean': 149, 'std': 23}
    }
    
    # Generate binary features
    n_males = int(n_samples * 0.7)  # 70% male
    sex = np.zeros(n_samples)
    sex[:n_males] = 1
    np.random.shuffle(sex)
    
    # Generate other categorical features
    chest_pain = np.random.randint(0, 4, size=n_samples)
    blood_sugar = (np.random.random(n_samples) < 0.15).astype(int)  # 15% have high blood sugar
    exercise = (np.random.random(n_samples) < 0.33).astype(int)  # 33% have exercise-induced angina
    
    # Create a dataframe with confounders
    data = {}
    for name, params in confounders.items():
        data[name] = np.random.normal(params['mean'], params['std'], n_samples)
    
    # Add binary and categorical features
    data['Sex'] = sex
    data['ChestPain'] = chest_pain
    data['BloodSugar'] = blood_sugar
    data['Exercise'] = exercise
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Generate propensity scores based on confounders and demographics
    propensity = -3.0  # Base propensity
    propensity += 0.05 * df['Age']  # Older patients more likely to receive treatment
    propensity += 0.01 * df['RestBP']  # Higher BP, more likely to receive treatment
    propensity += 0.005 * df['Cholesterol']  # Higher cholesterol, more likely treatment
    propensity += 0.5 * df['Sex']  # Males more likely to receive treatment
    propensity += 0.3 * df['ChestPain']  # More chest pain, more likely treatment
    propensity += 0.7 * df['BloodSugar']  # High blood sugar, more likely treatment
    propensity += 0.6 * df['Exercise']  # Exercise angina, more likely treatment
    
    # Convert to probability
    propensity = 1 / (1 + np.exp(-propensity))
    
    # Assign treatment
    treatment = (np.random.random(n_samples) < propensity).astype(int)
    df['Treatment'] = treatment
    
    # Generate outcome
    # Base outcome (higher is worse)
    outcome = 10.0
    
    # Effect of confounders on outcome
    outcome += 0.1 * df['Age']  # Age worsens outcome
    outcome -= 0.2 * df['MaxHR']  # Higher max HR is better
    outcome += 0.01 * df['Cholesterol']  # Higher cholesterol is worse
    outcome += 0.05 * df['RestBP']  # Higher BP is worse
    outcome += 1.5 * df['Exercise']  # Exercise angina is worse
    
    # Treatment effect (reduces outcome by 3 units)
    true_effect = -3.0
    outcome += true_effect * df['Treatment']
    
    # Add random noise
    outcome += np.random.normal(0, 2, n_samples)
    
    # Scale to reasonable range (0-100, higher is worse)
    outcome = 50 + 10 * outcome
    outcome = np.clip(outcome, 0, 100)
    
    df['Outcome'] = outcome
    
    # Add documentation
    metadata = {
        'dataset_name': 'synthetic_heart_disease',
        'n_samples': n_samples,
        'true_effect': true_effect,
        'treatment_col': 'Treatment',
        'outcome_col': 'Outcome',
        'potential_confounders': list(confounders.keys()) + ['Sex', 'ChestPain', 'BloodSugar', 'Exercise'],
        'description': 'Synthetic dataset mimicking heart disease data with treatment effect.'
    }
    
    return df, metadata

def generate_diabetes_dataset(n_samples=1000, random_state=42):
    """
    Generate a synthetic dataset mimicking diabetes treatment data.
    
    Features:
    - Age: age in years
    - Sex: 0 = female, 1 = male
    - BMI: body mass index
    - BP: blood pressure
    - S1-S4: lab test results
    - Treatment: if the patient received the treatment (0 = no, 1 = yes)
    - Outcome: disease progression metric (lower is better)
    """
    np.random.seed(random_state)
    
    # Define confounders
    confounders = {
        'Age': {'mean': 48, 'std': 5},
        'BMI': {'mean': 26.4, 'std': 4.3},
        'BP': {'mean': 94.6, 'std': 10.1}
    }
    
    # Lab test confounders
    for i in range(1, 5):
        confounders[f'S{i}'] = {'mean': 100 + i*20, 'std': 10 + i*2}
    
    # Generate binary features
    n_males = int(n_samples * 0.5)  # 50% male
    sex = np.zeros(n_samples)
    sex[:n_males] = 1
    np.random.shuffle(sex)
    
    # Create a dataframe with confounders
    data = {}
    for name, params in confounders.items():
        data[name] = np.random.normal(params['mean'], params['std'], n_samples)
    
    # Add binary features
    data['Sex'] = sex
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Generate propensity scores based on confounders
    propensity = -1.0  # Base propensity
    propensity += 0.03 * df['Age']  # Older patients more likely to receive treatment
    propensity += 0.1 * df['BMI']  # Higher BMI, more likely to receive treatment
    propensity += 0.02 * df['BP']  # Higher BP, more likely treatment
    propensity += 0.01 * df['S1']  # Lab results affect treatment
    propensity += 0.005 * df['S2']
    propensity += 0.002 * df['S3']
    propensity += 0.001 * df['S4']
    
    # Convert to probability
    propensity = 1 / (1 + np.exp(-propensity))
    
    # Assign treatment
    treatment = (np.random.random(n_samples) < propensity).astype(int)
    df['Treatment'] = treatment
    
    # Generate outcome (lower is better)
    # Base outcome
    outcome = 150.0
    
    # Effect of confounders on outcome
    outcome += 1.0 * df['Age']  # Age worsens outcome
    outcome += 2.5 * df['BMI']  # Higher BMI is worse
    outcome += 0.8 * df['BP']  # Higher BP is worse
    outcome += 0.05 * df['S1']  # Lab results affect outcome
    outcome += 0.05 * df['S2']
    outcome += 0.05 * df['S3']
    outcome += 0.05 * df['S4']
    
    # Treatment effect (reduces outcome by 25 units - improvement)
    true_effect = -25.0
    outcome += true_effect * df['Treatment']
    
    # Add random noise
    outcome += np.random.normal(0, 10, n_samples)
    
    # Scale to reasonable range
    outcome = np.clip(outcome, 50, 250)
    
    df['Outcome'] = outcome
    
    # Add documentation
    metadata = {
        'dataset_name': 'synthetic_diabetes',
        'n_samples': n_samples,
        'true_effect': true_effect,
        'treatment_col': 'Treatment',
        'outcome_col': 'Outcome',
        'potential_confounders': list(confounders.keys()) + ['Sex'],
        'description': 'Synthetic dataset mimicking diabetes treatment data with known effect.'
    }
    
    return df, metadata

def generate_generic_dataset(n_samples=1000, n_confounders=5, treatment_effect=2.0, random_state=42):
    """Generate a generic synthetic dataset with specified parameters."""
    data = generate_synthetic_data(
        n_samples=n_samples,
        binary_treatment=True,
        treatment_effect=treatment_effect,
        random_state=random_state
    )
    
    # Add documentation
    metadata = {
        'dataset_name': 'synthetic_generic',
        'n_samples': n_samples,
        'true_effect': treatment_effect,
        'treatment_col': 'treatment',
        'outcome_col': 'outcome',
        'potential_confounders': [col for col in data.columns if col not in ['treatment', 'outcome']],
        'description': 'Generic synthetic dataset with known treatment effect.'
    }
    
    return data, metadata

def main():
    """Main function to generate and save sample datasets."""
    parser = argparse.ArgumentParser(description='Generate sample datasets for causal inference')
    parser.add_argument('--dataset', type=str, default='all', 
                        choices=['all', 'heart', 'diabetes', 'generic'],
                        help='Dataset type to generate')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save datasets')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate and save datasets
    metadata_all = {}
    
    if args.dataset in ['all', 'heart']:
        print("Generating heart disease dataset...")
        heart_data, heart_metadata = generate_heart_disease_dataset(
            n_samples=args.n_samples,
            random_state=args.random_state
        )
        file_path = save_dataset(heart_data, 'heart_disease.csv', args.output_dir)
        metadata_all['heart_disease'] = heart_metadata
        metadata_all['heart_disease']['file_path'] = file_path
    
    if args.dataset in ['all', 'diabetes']:
        print("Generating diabetes dataset...")
        diabetes_data, diabetes_metadata = generate_diabetes_dataset(
            n_samples=args.n_samples,
            random_state=args.random_state
        )
        file_path = save_dataset(diabetes_data, 'diabetes.csv', args.output_dir)
        metadata_all['diabetes'] = diabetes_metadata
        metadata_all['diabetes']['file_path'] = file_path
    
    if args.dataset in ['all', 'generic']:
        print("Generating generic dataset...")
        generic_data, generic_metadata = generate_generic_dataset(
            n_samples=args.n_samples,
            treatment_effect=2.0,
            random_state=args.random_state
        )
        file_path = save_dataset(generic_data, 'generic.csv', args.output_dir)
        metadata_all['generic'] = generic_metadata
        metadata_all['generic']['file_path'] = file_path
    
    # Save metadata
    metadata_df = pd.DataFrame([
        {
            'dataset': name,
            'samples': meta['n_samples'],
            'treatment_column': meta['treatment_col'],
            'outcome_column': meta['outcome_col'],
            'true_effect': meta['true_effect'],
            'description': meta['description'],
            'file_path': meta.get('file_path', '')
        }
        for name, meta in metadata_all.items()
    ])
    
    metadata_path = os.path.join(args.output_dir, 'metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    main() 
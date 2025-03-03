import sys
import os

# Add the project root to the path to import local modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Import the dashboard from the visualizations module
from src.visualizations import dashboard
from src.data_processing.data_loader import load_dataset
from src.data_processing.preprocessing import preprocess_data

def main():
    # Example: Load data and preprocess
    data = load_dataset()
    processed_data = preprocess_data(data)
    
    # Example: Launch the dashboard
    dashboard.launch(processed_data)

if __name__ == "__main__":
    main()

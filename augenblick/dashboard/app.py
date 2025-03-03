import sys
import os

# Add the project root to the path to import local modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Import the dashboard from the visualizations module
from visualizations.dashboard import main

if __name__ == "__main__":
    # Launch the dashboard
    main()

from flask import Flask, request, jsonify
from flask_cors import CORS
from causal_model import run_analysis
import os
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the absolute path to the dataset
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ihdp_data.csv')

@app.route('/run-analysis', methods=['POST'])
def run_analysis_endpoint():
    """
    Expects a JSON body like:
    {
      "method": "PSM",
      "numNeighbors": 5,
      "trimWeights": 0.1
    }
    Returns the estimated ATE along with visualization data.
    """
    try:
        data = request.get_json()
        logger.info("Received request data: %s", data)
        logger.info("Using dataset path: %s", DATASET_PATH)
        
        if not os.path.exists(DATASET_PATH):
            logger.error("Dataset file not found at: %s", DATASET_PATH)
            return jsonify({'error': 'Dataset file not found'}), 404
        
        method = data.get('method', 'PSM')
        num_neighbors = int(data.get('numNeighbors', 5))  # Ensure integer
        trim_weights = float(data.get('trimWeights', 0.1))  # Ensure float
        
        logger.info("Running analysis with method=%s, num_neighbors=%d, trim_weights=%.2f",
                   method, num_neighbors, trim_weights)
        
        results = run_analysis(DATASET_PATH, method, num_neighbors=num_neighbors, trim_weights=trim_weights)
        
        logger.info("Generated results with keys: %s", results.keys())
        logger.info("Balance plot present: %s", 'balance_plot' in results)
        if 'balance_plot' in results:
            logger.info("Balance plot length: %d", len(results['balance_plot']))
        
        return jsonify(results)
    except ValueError as ve:
        logger.error("Value error: %s", str(ve))
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Invalid parameter value: {str(ve)}'}), 400
    except Exception as e:
        logger.error("Error in run_analysis: %s", str(e))
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

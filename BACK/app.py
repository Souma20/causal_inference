from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, Union, List
import uvicorn
import os
import logging
from causal_model import run_analysis, simulate_what_if_scenario, validate_model, load_ihdp_data

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="Causal Inference API")

# Configure CORS
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3002",
    "http://localhost:3003",
    "http://localhost:3004",
    "http://localhost:3005",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://127.0.0.1:3002",
    "http://127.0.0.1:3003",
    "http://127.0.0.1:3004",
    "http://127.0.0.1:3005"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define the absolute path to the dataset
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ihdp_data.csv')
logger.info(f"Dataset path configured as: {DATASET_PATH}")

# Pydantic models for request validation
class AnalysisRequest(BaseModel):
    method: str = "PSM"
    numNeighbors: int = 5
    trimWeights: float = 0.1
    include_diagnostics: bool = True

class ScenarioRequest(BaseModel):
    modified_covariates: Dict[str, float]
    method: str = "PSM"

class ValidationResponse(BaseModel):
    mean_cv_score: float
    std_cv_score: float
    ci_lower: float
    ci_upper: float

@app.get("/")
async def root():
    """
    Root endpoint for API verification
    """
    return {"message": "Causal Inference API is running"}

@app.post("/run-analysis")
async def run_analysis_endpoint(request: AnalysisRequest):
    """
    Run causal inference analysis with specified parameters
    """
    try:
        logger.info(f"Received analysis request: {request}")
        
        if not os.path.exists(DATASET_PATH):
            logger.error(f"Dataset file not found at: {DATASET_PATH}")
            raise HTTPException(status_code=404, detail=f"Dataset file not found at {DATASET_PATH}")
        
        logger.info("Loading dataset...")
        # Verify dataset can be loaded
        try:
            df = load_ihdp_data(DATASET_PATH)
            logger.info(f"Dataset loaded successfully with shape: {df.shape}")
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")
        
        logger.info(f"Running analysis with method={request.method}")
        results = run_analysis(
            DATASET_PATH,
            method=request.method,
            num_neighbors=request.numNeighbors,
            trim_weights=request.trimWeights
        )
        
        logger.info(f"Analysis completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error in run_analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulate-scenario")
async def simulate_scenario(request: ScenarioRequest):
    """
    Handle what-if scenario simulations
    """
    try:
        df = load_ihdp_data(DATASET_PATH)
        
        # Run simulation
        results = simulate_what_if_scenario(
            df=df,
            modified_covariates=request.modified_covariates,
            ps_model=None
        )
        
        # Add validation metrics
        validation = validate_model(df, method=request.method)
        results.update(validation)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    try:
        # Check if dataset exists
        dataset_exists = os.path.exists(DATASET_PATH)
        return {
            "status": "healthy",
            "dataset_exists": dataset_exists,
            "dataset_path": DATASET_PATH
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)

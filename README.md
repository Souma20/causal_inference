# Causal Inference Project

This project implements causal inference analysis with a FastAPI backend and React frontend.

## Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- npm or yarn

## Setup Instructions

### Backend Setup

```bash
# Navigate to backend directory
cd BACK

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn app:app --reload --port 5000
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd Front

# Install dependencies
npm install

# Start development server
npm run dev
```

## Environment Variables

### Backend (.env in BACK directory)
- DATASET_PATH: Path to the IHDP dataset
- PORT: Server port (default: 5000)
- DEBUG: Debug mode (True/False)
- CORS_ORIGINS: Allowed CORS origins

### Frontend (.env in Front directory)
- VITE_API_URL: Backend API URL
- VITE_NODE_ENV: Development environment

## Project Structure

```
causal_inference/
├── BACK/
│   ├── venv/
│   ├── app.py
│   ├── causal_model.py
│   ├── requirements.txt
│   └── ihdp_data.csv
├── Front/
│   ├── src/
│   │   ├── components/
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── public/
│   └── package.json
└── README.md
```

## Available Scripts

### Backend
- `uvicorn app:app --reload`: Start development server
- `pytest`: Run tests

### Frontend
- `npm run dev`: Start development server
- `npm run build`: Build for production
- `npm run preview`: Preview production build

## API Documentation

Once the backend server is running, visit:
- Swagger UI: http://localhost:5000/docs
- ReDoc: http://localhost:5000/redoc 
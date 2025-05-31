# LSTM MLOps Project for Stock Prediction

This project, an outcome of FIAP's Machine Learning Engineer graduate / post-graduate degree, centers on an LSTM-based stock prediction system that incorporates MLOps best practices. We're using FastAPI for serving predictions and MLflow for experiment tracking, model saves and deploy pipeline.

**My main emphasis was on implementing MLflow and setting up an API endpoint to serve the model with MLOps best practices.** It's no secret that predicting stock prices is really tough, given how unpredictable markets are. So, while the LSTM model itself is functional, it's definitely a work in progress and would need much more refinement (think hyperparameter tuning and better feature engineering) to be close to effective. **The focus was definitely on the MLOps pipeline, not on creating a perfect prediction model.**

## Project Overview

This project utilizes Deep Learning (LSTM) to predict stock prices, specifically for PETROBRAS (PETR4.SA). The system is containerized using Docker and employs MLOps practices for model tracking and serving.


## Architecture

```
├── api.py               # FastAPI endpoint for predictions
├── data/                # Data processing and storage
├── models/              # Trained model storage
├── mlruns/              # MLflow experiment tracking
├── scripts/             # Specific scripts to run processes
├── src/                 # Source code
│   ├── model/           # Model architecture
│   ├── training/        # Training procedures
│   └── utils/           # Utility functions
└── tests/               # Test suite
```

## Key Features

- **LSTM Model**: Deep Learning model for time series prediction
- **MLflow Integration**: Experiment tracking and model versioning
- **FastAPI Endpoint**: RESTful API for real-time predictions
- **Docker Support**: Containerization for reproducible deployments
- **Automated Testing**: Test suite for data and model validation

## Getting Started

### Prerequisites

- Docker
- Python 3.12+
- CUDA (optional, for GPU support)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/matheusbuniotto/LSTM_MLOps.git
cd LSTM_MLOps
```

2. Build and run with Docker:
```bash
make build
make run
```
2.1. Others make commands are avaliable, use make help to show
```bash
make help
make test
make stop 
make clean 
```

### Using the API

The prediction API is available at `http://localhost:8000/predict` with the following endpoints:

#### Predict Endpoint

```http
GET /predict?stock=PETR4.SA&window=30&start_date=2024-01-01
```

Parameters:
- `stock`: Stock symbol (default: PETR4.SA)
- `window`: Time window for prediction (default: 30)
- `start_date`: Start date for data collection (format: YYYY-MM-DD)

Example response:
```json
{
    "stock": "PETR4.SA",
    "predicted_value": 36.38
}
```

### MLflow Interface

MLflow UI is available at `http://localhost:8081` for:
- Experiment tracking
- Model versioning
- Performance metrics visualization
- Model registry

## Project Structure

### Core Components

1. **Data Processing (`data/`)**
   - `process_data.py`: Data collection and preprocessing using yahoo finance library
   - `PETRA_4.csv`: Historical stock data (fallback)

   ** The fallback was needed due to some rate limits erros when using yahoo finance. If theres no need for the fallback,
   the data will be collected directly from the yfinance.**  

2. **Model (`src/model/`)**
   - `lstm_model.py`: LSTM model architecture
   - Model configuration and training parameters

3. **Training (`src/training/`)**
   - `train.py`: Training loop and model optimization
   - Hyperparameter management

4. **Utils (`src/utils/`)**
   - `dataset.py`: Data loading and batching
   - `metrics.py`: Performance metrics
   - `model_manager.py`: Model saving and loading

5. **API (`api.py`)**
   - FastAPI implementation
   - Real-time prediction endpoint
   - Error handling and input validation

### Development Tools

- **Makefile**: Automation for common tasks
  - Docker management
  - Testing
  - Model training and prediction

## Development

### Running Tests

```bash
make test-docker
```

### Making Predictions

```bash
make predict-docker
```

### API Development

```bash
make predict-api-docker
```

## MLOps Features

- **Experiment Tracking**: All training runs are logged in MLflow
- **Model Registry**: Version control for trained models
- **Containerization**: Docker for consistent environments
- **API**: FastAPI for prediction endpoint

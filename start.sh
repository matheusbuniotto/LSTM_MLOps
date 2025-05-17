#!/bin/bash
set -e

echo "Running data processing script..."
python data/process_data.py || echo "[WARNING] data/process_data.py failed, continuing..."

#python scripts/run_train.py || echo "[WARNING] scripts/run_train.py failed, continuing..."

echo "Starting MLflow server..."
mlflow server --host 0.0.0.0 --port 8081 &
MLFLOW_PID=$!

echo "Starting FastAPI server..."
uvicorn api:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Wait for any process to exit
wait -n

# Kill remaining processes
kill $MLFLOW_PID $FASTAPI_PID 2>/dev/null || true
wait
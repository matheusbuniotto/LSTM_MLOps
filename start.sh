#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Running data processing script..."
python data/process_data.py || echo "[WARNING] data/process_data.py failed, continuing..."
python scripts/run_train.py || echo "[WARNING] scripts/run_train.py failed, continuing..."

echo "Starting MLflow server..."
exec mlflow server --host 0.0.0.0 --port 8081

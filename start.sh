#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Running data processing script..."
python data/process_data.py
python scripts/run_train.py

# Check if the script ran successfully (optional but good practice)
if [ $? -ne 0 ]; then
  echo "Data processing script failed. Exiting."
  exit 1
fi

echo "Starting MLflow server..."
# Using 'exec' replaces the current bash process with the mlflow server process.
# This is a good practice as it passes signals correctly.
# IMPORTANT: Changed host to 0.0.0.0 so it's accessible from outside the container.
exec mlflow server --host 0.0.0.0 --port 8080

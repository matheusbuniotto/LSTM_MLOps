#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Running data processing script..."
python data/process_data.py
python scripts/test.py

# Check if the script ran successfully (optional but good practice)
if [ $? -ne 0 ]; then
  echo "Data processing script failed. Exiting."
  exit 1
fi

echo "Starting MLflow server..."
exec mlflow server --host 0.0.0.0 --port 8081

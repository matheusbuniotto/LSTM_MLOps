import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow.pytorch
from mlflow import MlflowClient

client = MlflowClient()
mlflow.set_tracking_uri(uri="http://0.0.0.0:8081")
model_name = "lstm-2000-epochs"
model_version_alias = "champion"

try:
    # Get information about the model
    model_info = client.get_model_version_by_alias(model_name, model_version_alias)
    print(f"Loading champion model (version {model_info.version})")
    print(f"Model tags: {model_info.tags}")
    
    # Load the model using the alias
    model_uri = f"models:/{model_name}@{model_version_alias}"
    # Add map_location to handle models saved on CUDA but loaded in CPU-only environments
    model = mlflow.pytorch.load_model(model_uri, map_location=torch.device('cpu'))
    
    print(f"Successfully loaded model: {model}")
except Exception as e:
    print(f"Error loading champion model: {e}")

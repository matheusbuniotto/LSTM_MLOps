import os
import sys
import time
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import mlflow.pytorch
from mlflow import MlflowClient

def run_test():
    start_time = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting model test...")
    
    # Connect to MLflow
    try:
        client = MlflowClient()
        mlflow_uri = "http://0.0.0.0:8081"
        if "MLFLOW_TRACKING_URI" in os.environ:
            mlflow_uri = os.environ["MLFLOW_TRACKING_URI"]
        
        print(f"Connecting to MLflow at: {mlflow_uri}")
        mlflow.set_tracking_uri(uri=mlflow_uri)
        model_name = "lstm-2000-epochs"
        model_version_alias = "champion"
        
        # Get information about the model
        model_info = client.get_model_version_by_alias(model_name, model_version_alias)
        print(f"Found champion model (version {model_info.version})")
        print(f"Model tags: {model_info.tags}")
        
        # Load the model using the alias
        model_uri = f"models:/{model_name}@{model_version_alias}"
        # Add map_location to handle models saved on CUDA but loaded in CPU-only environments
        model = mlflow.pytorch.load_model(model_uri, map_location=torch.device('cpu'))
        
        # Basic model validation
        if not isinstance(model, torch.nn.Module):
            raise ValueError("Loaded model is not a valid PyTorch model")
            
        # Check if model has expected structure (LSTM layers)
        has_lstm = any(isinstance(module, torch.nn.LSTM) for module in model.modules())
        if not has_lstm:
            raise ValueError("Model does not contain expected LSTM layers")
            
        # Print model structure
        print(f"Model structure: {model}")
        
        # Run a simple inference test with random data
        test_input = torch.randn(1, 7, 1)  # Batch size 1, sequence length 7, feature dim 1
        try:
            with torch.no_grad():
                output = model(test_input)
            print(f"Test inference successful, output shape: {output.shape}")
        except Exception as e:
            raise RuntimeError(f"Inference test failed: {e}")

        elapsed_time = time.time() - start_time
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Test completed in {elapsed_time:.2f} seconds")
        print("\n✅ TEST RESULT: OK - Model is valid and operational")
        return True
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Test failed after {elapsed_time:.2f} seconds")
        print(f"❌ TEST RESULT: NOT OK - {str(e)}")
        return False

if __name__ == "__main__":
    success = run_test()
    # Exit with appropriate code for CI/CD pipelines
    sys.exit(0 if success else 1)

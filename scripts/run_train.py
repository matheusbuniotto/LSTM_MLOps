import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from data.process_data import DataCollector
from src.model.lstm_model import LSTMModel
from src.training.train import evaluate_model, train_model
from src.utils.dataset import TimeSeriesDataset
from src.utils.metrics import root_mean_squared_error
from src.utils.model_manager import ModelManager

collector = DataCollector("PETR4.SA")
collector.get_data(start_date="2023-01-01")
collector.split_data(test_size=0.2, window=7)
collector.standard_scale()

train_dataset = TimeSeriesDataset(collector.X_train, collector.y_train)
test_dataset = TimeSeriesDataset(collector.X_test, collector.y_test)

# Model
model_args = {
    "input_size": 1,
    "hidden_size": 50,
    "num_layers": 5,
    "output_size": 1,
    "dropout": 0.1,
}

model = LSTMModel(**model_args)
model = train_model(model, train_dataset, test_dataset, num_epochs=2500)

predictions, actuals = evaluate_model(model, test_dataset)
rmse = root_mean_squared_error(np.vstack(actuals), np.vstack(predictions))

print("Test RMSE:", rmse)

ModelManager.save_model(model, "models/lstm_petra.pth", model_args)

# ML FLow
#
import mlflow
from mlflow.models import infer_signature

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://0.0.0.0:8081")
# Create a new MLflow Eperiment
mlflow.set_experiment("LSTM-PETRA")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(model_args)

    # Log the loss metric
    mlflow.log_metric("RMSE", rmse)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "LSTM model for PETRA4")

    # Infer the model signature
    signature = infer_signature(collector.X_train, predictions)

    # Log the model
    model_info = mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="mlartifacts",
        signature=signature,
        input_example=collector.X_train,
        registered_model_name="lstm-2000-epochs",
    )

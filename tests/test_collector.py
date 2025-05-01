import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from data.process_data import DataCollector
from src.utils.dataset import TimeSeriesDataset
from src.utils.model_manager import ModelManager
from src.model.lstm_model import LSTMModel
from src.training.train import evaluate_model
from src.utils.metrics import root_mean_squared_error, mean_absolute_error

collector = DataCollector("PETR4.SA")
collector.get_data(start_date="2023-01-01")
collector.split_data(test_size=0.1, window=20)

collector.standard_scale()

print(collector)

test_dataset = TimeSeriesDataset(collector.X_test, collector.y_test)
print(f"Test: {test_dataset}")

# Load trained model
model = ModelManager.load_model(LSTMModel, "models/lstm_petra.pth")

# Evaluate
predictions, actuals = evaluate_model(model, test_dataset)

# Stack
predictions = np.vstack(predictions)  # shape (N, 1)
actuals = np.vstack(actuals).reshape(-1, 1)  # shape (N, 1)

# Confirm shape
print("predictions shape:", predictions.shape)
print("actuals shape:", actuals.shape)

# Inverse transform
predictions_original = collector.target_scaler.inverse_transform(predictions)
actuals_original = collector.target_scaler.inverse_transform(actuals)

print("pred original:", predictions_original)
print("actual original:", actuals_original)
# Metrics
rmse = root_mean_squared_error(actuals_original, predictions_original)
mae = mean_absolute_error(actuals_original, predictions_original)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")

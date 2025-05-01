import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model.lstm_model import LSTMModel
from src.utils.dataset import TimeSeriesDataset
from src.utils.model_manager import ModelManager
from src.utils.metrics import root_mean_squared_error, mean_absolute_error
from data.process_data import DataCollector
from src.training.train import evaluate_model
import numpy as np
import matplotlib.pyplot as plt

collector = DataCollector("PETRA4.SA")
collector.get_data(start_date="2023-01-01")
collector.split_data(test_size=0.1, window=20)
collector.standard_scale()

# Prepare test dataset
test_dataset = TimeSeriesDataset(collector.X_test, collector.y_test)


model = ModelManager.load_model(LSTMModel, "models/lstm_petra.pth")


# Predict
predictions, actuals = evaluate_model(model, test_dataset)

# Reshape
predictions = np.vstack(predictions)
actuals = np.vstack(actuals)

# Inverse transform
predictions_original = collector.scaler.inverse_transform(predictions)
actuals_original = collector.scaler.inverse_transform(actuals)

# Metrics
rmse = root_mean_squared_error(actuals_original, predictions_original)
mae = mean_absolute_error(actuals_original, predictions_original)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")

# Optional: Plot predictions vs real values
plt.figure(figsize=(10, 6))
plt.plot(actuals_original, label="Actual")
plt.plot(predictions_original, label="Predicted")
plt.legend()
plt.title("Stock Price Prediction vs Actual")
plt.show()

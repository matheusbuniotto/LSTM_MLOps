import numpy as np

from data.process_data import DataCollector
from src.model.lstm_model import LSTMModel
from src.training.train import evaluate_model, train_model
from src.utils.dataset import TimeSeriesDataset
from src.utils.metrics import root_mean_squared_error
from src.utils.model_manager import ModelManager

collector = DataCollector("PETR4.SA")
collector.get_data(start_date="2023-01-01")
collector.split_data(test_size=0.1, window=20)
collector.standard_scale()

train_dataset = TimeSeriesDataset(collector.X_train, collector.y_train)
test_dataset = TimeSeriesDataset(collector.X_test, collector.y_test)

model = LSTMModel(input_size=1, hidden_size=50, num_layers=3, output_size=1)
model = train_model(model, train_dataset, test_dataset, num_epochs=2500)

predictions, actuals = evaluate_model(model, test_dataset)

rmse = root_mean_squared_error(np.vstack(actuals), np.vstack(predictions))

print("Test RMSE:", rmse)

ModelManager.save_model(model, "src/model/lstm_petra.pth")

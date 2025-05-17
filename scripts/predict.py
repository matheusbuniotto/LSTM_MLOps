import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import torch
from datetime import datetime

from data.process_data import DataCollector
from src.model.lstm_model import LSTMModel
from src.utils.model_manager import ModelManager

def predict_next_day(stock="PETR4.SA", window=30, start_date="2024-01-01"):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting prediction for stock: {stock}")
    
    # Initialize data collector and fetch data
    collector = DataCollector(stock)
    collector.get_data(start_date=start_date)
    
    # Split data (needed before scaling)
    collector.split_data(test_size=0.1, window=window)
    
    # Apply scaling (this function is assumed to scale and store the scaler in collector.target_scaler)
    collector.standard_scale()
    
    # Check that the raw data is loaded
    if collector.data is None:
        print("DataCollector missing 'data' attribute with fetched prices")
        return

    # Extract only the close prices as input for prediction
    close_prices = np.array(collector.data["close"])
    if close_prices.shape[0] < window:
        print(f"Not enough data to form a window of size {window}")
        return

    # Get the last `window` number of close prices
    latest_window = close_prices[-window:]
    # Reshape to (window, 1) for model input
    input_seq = latest_window.reshape(window, 1)
    
    # Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelManager.load_model(LSTMModel, "models/lstm_petra.pth")
    model = model.to(device)
    model.eval()
    
    # Convert input sequence to torch tensor with batch dimension
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction_scaled = model(input_tensor)
    
    # Get the prediction value (assumed shape is (1, 1))
    pred_value = prediction_scaled.cpu().numpy().flatten()[0]
    
    # Inverse transform to original scale using the stored target scaler
    pred_original = collector.target_scaler.inverse_transform(np.array([[pred_value]]))[0, 0]
    
    print(f"Predicted next day value for {stock}: {pred_original:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict next day stock price")
    parser.add_argument("--stock", type=str, default="PETR4.SA", help="Stock symbol")
    parser.add_argument("--window", type=int, default=30, help="Window size for prediction")
    parser.add_argument("--start_date", type=str, default="2024-01-01", help="Start date for fetching data (YYYY-MM-DD)")
    args = parser.parse_args()
    
    predict_next_day(stock=args.stock, window=args.window, start_date=args.start_date)
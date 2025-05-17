from fastapi import FastAPI, HTTPException
import numpy as np
import torch
from data.process_data import DataCollector
from src.model.lstm_model import LSTMModel
from src.utils.model_manager import ModelManager

app = FastAPI()

@app.get("/predict")
def predict(stock: str = "PETR4.SA", window: int = 30, start_date: str = "2024-01-01"):
    try:
        # Initialize data collector and fetch data
        collector = DataCollector(stock)
        collector.get_data(start_date=start_date)
        collector.split_data(test_size=0.1, window=window)
        collector.standard_scale()
        
        if collector.data is None:
            raise HTTPException(status_code=500, detail="DataCollector missing 'data' attribute with fetched prices")
        
        close_prices = np.array(collector.data["close"])
        if close_prices.shape[0] < window:
            raise HTTPException(status_code=400, detail=f"Not enough data to form a window of size {window}")
        
        # Get the last `window` number of close prices and reshape
        latest_window = close_prices[-window:]
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
        
        # Get prediction value and inverse transform to original scale
        pred_value = prediction_scaled.cpu().numpy().flatten()[0]
        pred_original = float(collector.target_scaler.inverse_transform(np.array([[pred_value]]))[0, 0])
        
        return {"stock": stock, "predicted_value": pred_original}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
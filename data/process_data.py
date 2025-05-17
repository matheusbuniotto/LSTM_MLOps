import os
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import requests 

class DataCollector:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = None
        self.X_train = None
        self.X_test = None

    def get_data(self, start_date=None, end_date=None, session=None):
        if session is None:
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/70.0.3538.77 Safari/537.36"
            })
        try:
            stock = yf.Ticker(self.ticker)
            stock_collected = stock.history(period="max", auto_adjust=True)
            stock_collected.index = stock_collected.index.tz_localize(None)

            data = pd.DataFrame({
                "date": stock_collected.index,
                "close": stock_collected["Close"].values,
            })
            if start_date:
                start_date = pd.to_datetime(start_date)
                data = data[data["date"] >= start_date]
            if end_date:
                end_date = pd.to_datetime(end_date)
                data = data[data["date"] <= end_date]
            self.data = data.reset_index(drop=True)
        except Exception as e:
            print(f"Error fetching data from yfinance: {e}")
            print("Falling back to PETRA_4.csv data for prediction.")
            fallback_path = os.path.join("data", "raw", "PETRA_4.csv")
            self.data = pd.read_csv(fallback_path, parse_dates=["date"])

    def split_data(self, test_size=0.2, window=30):
        if self.data is None:
            raise ValueError("Data not loaded. Please call get_data() first.")
        data = self.data.copy()
        closes = data["close"].values

        X, y = [], []
        for i in range(window, len(closes)):
            X.append(closes[i - window : i])
            y.append(closes[i])

        X = np.array(X)
        y = np.array(y)

        split_index = int((1 - test_size) * len(X))
        self.X_train, self.X_test = X[:split_index], X[split_index:]
        self.y_train, self.y_test = y[:split_index], y[split_index:]

    def standard_scale(self):
        if self.X_train is None or self.X_test is None:
            raise ValueError("Data not split yet. Please call split_data() first.")

        train_samples, train_nx = self.X_train.shape
        test_samples, test_nx = self.X_test.shape

        # Reshape X for scaling
        self.X_train = self.X_train.reshape((train_samples, train_nx, 1))
        self.X_test = self.X_test.reshape((test_samples, test_nx, 1))

        # Scale X
        self.scaler = StandardScaler()
        X_train_flat = self.X_train.reshape((train_samples, -1))
        X_test_flat = self.X_test.reshape((test_samples, -1))

        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        X_test_scaled = self.scaler.transform(X_test_flat)

        self.X_train = X_train_scaled.reshape((train_samples, train_nx, 1))
        self.X_test = X_test_scaled.reshape((test_samples, test_nx, 1))

        # Scale y (target)
        self.target_scaler = StandardScaler()
        self.y_train = self.target_scaler.fit_transform(self.y_train.reshape(-1, 1))
        self.y_test = self.target_scaler.transform(self.y_test.reshape(-1, 1))

    def save_csv(self, file_path=None):
        if self.data is None:
            raise ValueError("No data to save. Please call get_data() first.")
        if file_path is None:
            file_path = os.path.join("data", "raw", f"{self.ticker}_data.csv")
        self.data.to_csv(file_path, index=False)

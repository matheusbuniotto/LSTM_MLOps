import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


class DataCollector:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = None
        self.X_train = None
        self.X_test = None

    def get_data(self, start_date=None, end_date=None):
        stock = yf.Ticker(self.ticker)
        stock_collected = stock.history(period="max", auto_adjust=True)
        stock_collected.index = stock_collected.index.tz_localize(None)
        data = pd.DataFrame(
            {"date": stock_collected.index, "close": stock_collected["Close"].values}
        )
        if start_date:
            start_date = pd.to_datetime(start_date)
            data = data[data["date"] >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date)
            data = data[data["date"] <= end_date]
        self.data = data.reset_index(drop=True)

    def split_data(self, test_size=0.2, window=20):
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

        self.X_train = self.X_train.reshape((train_samples, train_nx, 1))
        self.X_test = self.X_test.reshape((test_samples, test_nx, 1))

        train_samples, train_nx, train_ny = self.X_train.shape
        test_samples, test_nx, test_ny = self.X_test.shape

        X_train_flat = self.X_train.reshape((train_samples, train_nx * train_ny))
        X_test_flat = self.X_test.reshape((test_samples, test_nx * test_ny))

        self.scaler = StandardScaler().fit(
            X_train_flat
        )  # armazenando o scaler para uso posterior
        X_train_scaled = self.scaler.transform(X_train_flat)
        X_test_scaled = self.scaler.transform(X_test_flat)

        self.X_train = X_train_scaled.reshape((train_samples, train_nx, train_ny))
        self.X_test = X_test_scaled.reshape((test_samples, test_nx, test_ny))

    def save_csv(self, file_path=None):
        if self.data is None:
            raise ValueError("No data to save. Please call get_data() first.")
        if file_path is None:
            file_path = f"data/raw/{self.ticker}_data.csv"
        self.data.to_csv(file_path, index=False)


collector = DataCollector("AAPL")
collector.get_data(start_date="2020-01-01")
collector.split_data(test_size=0.1, window=25)
collector.standard_scale()

print("X_train:", collector.X_train.shape)
print("y_train:", collector.y_train.shape)
print("X_test:", collector.X_test.shape)
print("y_test:", collector.y_test.shape)

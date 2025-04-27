import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


class DataCollector:
    def __init__(self, ticker):
        self.ticker = ticker

    def get_data(self):
        ticker_name = self.ticker
        stock = yf.Ticker(ticker_name)
        stock_collected = stock.history(period="max", auto_adjust=True)
        stock_collected.index = stock_collected.index.tz_localize(None)
        stock_data = pd.DataFrame()
        stock_data["date"] = stock_collected.index
        stock_data["close"] = stock_collected["Close"].values
        return stock_collected, stock_data

    def scale_data(self, data):
        scaler = MinMaxScaler()
        data["close_scaled"] = scaler.fit_transform(data[["close"]])
        return data

    def filter_data(self, data, start_date, end_date=None):
        start_date = pd.to_datetime(start_date)
        if end_date is not None:
            end_date = pd.to_datetime(end_date)
            mask = (data["date"] >= start_date) & (data["date"] <= end_date)
        else:
            mask = data["date"] >= start_date
        return data.loc[mask].reset_index(drop=True)

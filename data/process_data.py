import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


class DataCollector:
    "coleta e pprocessa dados de um ticker"

    def __init__(self, ticker):
        self.ticker = ticker
        self.data = None

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

    def scale_data(self):
        if self.data is None:
            raise ValueError("Dados não encontrados. Use get_data() antes.")
        scaler = MinMaxScaler()
        self.data["close_scaled"] = scaler.fit_transform(self.data[["close"]])

    def save_csv(self, file_path=None):
        if self.data is None:
            raise ValueError("Erro ao salvar. Use get_data() antes.")
        if file_path is None:
            file_path = f"data/raw/{self.ticker}_data.csv"
        self.data.to_csv(file_path, index=False)


collector = DataCollector("AAPL")
collector.get_data(start_date="2023-01-01")
collector.scale_data()
collector.save_csv()

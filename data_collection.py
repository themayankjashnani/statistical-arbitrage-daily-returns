import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)
    return hist

historical_data = fetch_stock_data('AAPL', '2010-01-01', '2025-12-31')
data = historical_data[['Open', 'High', 'Low', 'Close', 'Volume']]
data.to_csv('data/stock_data.csv')
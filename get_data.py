import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime

# Paste your keys here
API_KEY = "PKBVRHYNY7IN5Z6GWHFZJM3FYO"
SECRET_KEY = "9PsNrE5tmXyxor1bCPcg9Sy2ZFN2FbPT7CZodNqy6uiM"

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

request = StockBarsRequest(
    symbol_or_symbols="SPY",
    timeframe=TimeFrame.Minute,
    start=datetime(2023, 1, 1),
    end=datetime(2026, 9, 1),
)

print("Downloading SPY minute data...")
bars = client.get_stock_bars(request)
df = bars.df

# Clean up
df = df.reset_index()
df = df[df['symbol'] == 'SPY']
df = df.set_index('timestamp')
df.index = df.index.tz_convert('America/New_York')
df = df[['open', 'high', 'low', 'close', 'volume']]

print(f"Downloaded {len(df):,} bars")
print(df.tail())

df.to_parquet('SPY_1min.pq')
print("Saved to SPY_1min.pq")
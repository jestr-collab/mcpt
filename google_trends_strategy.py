import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from bar_permute import get_permutation
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pytrends.request import TrendReq

# ── Pull Google Trends data ──────────────────────────────────────────────────
print("Fetching Google Trends...")
pytrends = TrendReq(hl='en-US', tz=360)

keywords = ['stock market crash', 'recession', 'stock market', 'buy stocks']
pytrends.build_payload(keywords, timeframe='today 5-y', geo='US')
trends = pytrends.interest_over_time()

if 'isPartial' in trends.columns:
    trends = trends.drop(columns=['isPartial'])

print(f"Trends shape: {trends.shape}")
print(trends.tail())

# ── Pull SPY daily data ──────────────────────────────────────────────────────
import yfinance as yf
spy = yf.download('SPY', period='5y', interval='1d')
spy.columns = spy.columns.get_level_values(0)
spy.columns = [c.lower() for c in spy.columns]
spy.index = pd.to_datetime(spy.index)
if spy.index.tz is not None:
    spy.index = spy.index.tz_localize(None)

# ── Merge ────────────────────────────────────────────────────────────────────
# Trends is weekly — forward fill to daily
trends.index = pd.to_datetime(trends.index)
if trends.index.tz is not None:
    trends.index = trends.index.tz_localize(None)

trends_daily = trends.reindex(spy.index, method='ffill')
df = spy.join(trends_daily, how='inner').dropna()
print(f"\nMerged shape: {df.shape}")
print(df.tail())

df.to_parquet('SPY_trends.pq')
print("Saved to SPY_trends.pq")
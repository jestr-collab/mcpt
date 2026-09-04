import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_tree_v2(ohlc: pd.DataFrame):
    log_c = np.log(ohlc['close'])

    diff6 = log_c.diff(6)
    diff24 = log_c.diff(24)
    diff72 = log_c.diff(72)
    diff168 = log_c.diff(168)
    diff336 = log_c.diff(336)
    volatility = log_c.diff().rolling(24).std()
    rsi = compute_rsi(ohlc['close'], 14) / 100
    hour = pd.Series(ohlc.index.hour, index=ohlc.index) / 23
    day_of_week = pd.Series(ohlc.index.dayofweek, index=ohlc.index) / 6

    target = np.sign(log_c.diff(24).shift(-24))
    target = (target + 1) / 2

    dataset = pd.concat([
        diff6, diff24, diff72, diff168, diff336,
        volatility, rsi, hour, day_of_week, target
    ], axis=1)
    dataset.columns = [
        'diff6', 'diff24', 'diff72', 'diff168', 'diff336',
        'volatility', 'rsi', 'hour', 'day_of_week', 'target'
    ]

    train_data = dataset.dropna()
    feature_cols = [c for c in train_data.columns if c != 'target']
    train_x = train_data[feature_cols].to_numpy()
    train_y = train_data['target'].astype(int).to_numpy()

    model = DecisionTreeClassifier(min_samples_leaf=5, random_state=69)
    model.fit(train_x, train_y)
    return model, feature_cols

def tree_strategy_v2(ohlc: pd.DataFrame, model, feature_cols):
    log_c = np.log(ohlc['close'])

    diff6 = log_c.diff(6)
    diff24 = log_c.diff(24)
    diff72 = log_c.diff(72)
    diff168 = log_c.diff(168)
    diff336 = log_c.diff(336)
    volatility = log_c.diff().rolling(24).std()
    rsi = compute_rsi(ohlc['close'], 14) / 100
    hour = pd.Series(ohlc.index.hour, index=ohlc.index) / 23
    day_of_week = pd.Series(ohlc.index.dayofweek, index=ohlc.index) / 6

    dataset = pd.concat([
        diff6, diff24, diff72, diff168, diff336,
        volatility, rsi, hour, day_of_week
    ], axis=1)
    dataset.columns = feature_cols
    dataset = dataset.dropna()

    pred = model.predict(dataset.to_numpy())
    pred = pd.Series(pred, index=dataset.index)
    pred = pred.reindex(ohlc.index)

    signal = np.where(pred > 0, 1, -1)
    signal = pd.Series(signal, index=ohlc.index)

    r = log_c.diff().shift(-1)
    rets = signal * r
    pf = rets[rets > 0].sum() / rets[rets < 0].abs().sum()
    return signal, pf
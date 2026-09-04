import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import logging
import json
import os

# ── Config ────────────────────────────────────────────────────────────────────

API_KEY    = "PKBVRHYNY7IN5Z6GWHFZJM3FYO"
SECRET_KEY = "9PsNrE5tmXyxor1bCPcg9Sy2ZFN2FbPT7CZodNqy6uiM"
PAPER      = True  # Always True until you're ready for real money

SYMBOL     = "GLD"
SHARES     = 1        # Start with 1 share — conservative
LOG_FILE   = "paper_trade_log.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('paper_trade.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Triple Barrier + Features (same as research) ──────────────────────────────

def get_daily_vol(close, span0=20):
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0-1], index=close.index[close.shape[0]-df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    return df0.ewm(span=span0).std()

def build_features(df):
    log_c = np.log(df['close'])
    f = pd.DataFrame(index=df.index)
    f['mom5']      = log_c.diff(5)
    f['mom20']     = log_c.diff(20)
    f['mom60']     = log_c.diff(60)
    f['overnight'] = np.log(df['open'] / df['close'].shift(1))
    f['intraday']  = np.log(df['close'] / df['open'])
    f['vol20']     = log_c.diff().rolling(20).std()
    if 'dxy' in df.columns:
        f['dxy_mom5']           = np.log(df['dxy']).diff(5)
        f['dxy_mom20']          = np.log(df['dxy']).diff(20)
        f['dxy_change']         = np.log(df['dxy']).diff(1)
        f['gc_dxy_ratio']       = np.log(df['close'] / df['dxy'])
        f['gc_dxy_mom10']       = f['gc_dxy_ratio'].diff(10)
        f['gc_dxy_divergence']  = log_c.diff(5) + np.log(df['dxy']).diff(5)
    if 'vix' in df.columns:
        f['vix_level']    = df['vix'] / 100
        f['vix_change']   = df['vix'].pct_change().clip(-5, 5)
        f['vix_above20']  = (df['vix'] > 20).astype(int)
        f['vix_ma_ratio'] = df['vix'] / df['vix'].rolling(20).mean()
        f['vix_spike_25'] = (df['vix'] > 25).astype(int)
        f['vix_spike_30'] = (df['vix'] > 30).astype(int)
        f['vix_5d_change']= df['vix'].pct_change(5).clip(-5, 5)
    if 'tlt' in df.columns:
        f['tlt_mom5']      = np.log(df['tlt']).diff(5)
        f['tlt_mom20']     = np.log(df['tlt']).diff(20)
        f['tlt_change']    = np.log(df['tlt']).diff(1)
        gc_ret             = log_c.diff()
        tlt_ret            = np.log(df['tlt']).diff()
        f['gc_tlt_corr20'] = gc_ret.rolling(20).corr(tlt_ret)
    month = pd.to_datetime(df.index).month
    f['is_september'] = (month == 9).astype(int)
    f['is_november']  = (month == 11).astype(int)
    f['is_january']   = (month == 1).astype(int)
    f['is_june']      = (month == 6).astype(int)
    f['month']        = month / 12
    f['day_of_week']  = df.index.dayofweek / 4
    return f.dropna()

# ── Train model on historical data ───────────────────────────────────────────

def train_model():
    log.info("Loading historical data for model training...")
    df = pd.read_parquet('GLD_macro.pq')
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index.date)
    df.index.name = None
    df = df[df['volume'] > 0].dropna(subset=['close', 'dxy', 'vix'])

    close = df['close'].copy()
    close.index.name = None

    daily_vol = get_daily_vol(close)
    daily_vol.index.name = None

    # Simple labeling — next 5 day return direction
    log_c = np.log(close)
    target = np.sign(log_c.shift(-5) - log_c)
    target = (target > 0).astype(int)

    features = build_features(df)
    features.index.name = None

    common = features.index.intersection(target.dropna().index)
    X = features.loc[common]
    y = target.loc[common]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    train_acc = (model.predict(X_scaled) == y).mean()
    log.info(f"Model trained on {len(X)} samples, accuracy: {train_acc:.3f}")

    return model, scaler, features.columns.tolist()

# ── Get latest market data ────────────────────────────────────────────────────

def get_latest_data():
    log.info("Fetching latest market data...")
    end = datetime.now()
    start = end - timedelta(days=120)

    gld = yf.download('GLD', start=start, end=end, interval='1d', progress=False)
    gld.columns = gld.columns.get_level_values(0)
    gld.columns = [c.lower() for c in gld.columns]
    gld.index = pd.to_datetime(gld.index.date)
    gld.index.name = None

    dxy = yf.download('DX-Y.NYB', start=start, end=end, interval='1d', progress=False)
    dxy = dxy['Close'].squeeze()
    dxy.index = pd.to_datetime(dxy.index.date)
    dxy.index.name = None

    vix = yf.download('^VIX', start=start, end=end, interval='1d', progress=False)
    vix = vix['Close'].squeeze()
    vix.index = pd.to_datetime(vix.index.date)
    vix.index.name = None

    tlt = yf.download('TLT', start=start, end=end, interval='1d', progress=False)
    tlt = tlt['Close'].squeeze()
    tlt.index = pd.to_datetime(tlt.index.date)
    tlt.index.name = None

    gld['dxy'] = dxy
    gld['vix'] = vix
    gld['tlt'] = tlt

    gld = gld.dropna(subset=['close', 'dxy', 'vix'])
    log.info(f"Latest data: {len(gld)} bars, last date: {gld.index[-1]}")
    return gld

# ── Generate signal ───────────────────────────────────────────────────────────

def generate_signal(model, scaler, feature_cols, df):
    features = build_features(df)
    features.index.name = None

    if len(features) == 0:
        log.warning("No features generated")
        return 0, 0.0

    latest = features.iloc[[-1]]
    available = [f for f in feature_cols if f in latest.columns]
    X = latest[available].to_numpy()

    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]
    confidence = max(prob)

    signal = 1 if pred == 1 else -1
    log.info(f"Signal: {'LONG' if signal == 1 else 'SHORT'} | Confidence: {confidence:.3f}")
    return signal, confidence

# ── Execute paper trade ───────────────────────────────────────────────────────

def execute_trade(signal, confidence, current_price):
    client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)

    # Get current position
    try:
        position = client.get_open_position(SYMBOL)
        current_qty = float(position.qty)
        current_side = 'long' if current_qty > 0 else 'short'
    except Exception:
        current_qty = 0
        current_side = None

    log.info(f"Current position: {current_qty} shares ({current_side})")

    # Only trade if confidence above threshold
    if confidence < 0.55:
        log.info(f"Confidence too low ({confidence:.3f}) — no trade")
        return None

    # Determine action
    if signal == 1 and current_side != 'long':
        # Close short if exists, go long
        if current_side == 'short':
            order = MarketOrderRequest(
                symbol=SYMBOL,
                qty=abs(current_qty),
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            client.submit_order(order)
            log.info(f"Closed short position")

        order = MarketOrderRequest(
            symbol=SYMBOL,
            qty=SHARES,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        result = client.submit_order(order)
        log.info(f"LONG {SHARES} shares of {SYMBOL} at ~${current_price:.2f}")
        return result

    elif signal == -1 and current_side != 'short':
        # Close long if exists
        if current_side == 'long':
            order = MarketOrderRequest(
                symbol=SYMBOL,
                qty=abs(current_qty),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            client.submit_order(order)
            log.info(f"Closed long position")

        log.info(f"SHORT signal — skipping (ETFs are hard to short in paper)")
        return None

    else:
        log.info("No position change needed")
        return None

# ── Log trade to file ─────────────────────────────────────────────────────────

def log_trade(date, signal, confidence, price, action):
    entry = {
        'date': str(date),
        'signal': signal,
        'confidence': round(confidence, 4),
        'price': round(price, 2),
        'action': action
    }

    trades = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            trades = json.load(f)

    trades.append(entry)

    with open(LOG_FILE, 'w') as f:
        json.dump(trades, f, indent=2)

    log.info(f"Trade logged: {entry}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 50)
    log.info("GLD Paper Trading System")
    log.info("=" * 50)

    # Train model
    model, scaler, feature_cols = train_model()

    # Get latest data
    df = get_latest_data()

    # Generate signal
    signal, confidence = generate_signal(model, scaler, feature_cols, df)

    # Current price
    current_price = df['close'].iloc[-1]
    today = df.index[-1]

    log.info(f"Date: {today} | Price: ${current_price:.2f}")
    log.info(f"Signal: {'LONG' if signal == 1 else 'SHORT'} | Confidence: {confidence:.3f}")

    # Execute
    action = "HOLD"
    if confidence >= 0.55:
        result = execute_trade(signal, confidence, current_price)
        if result:
            action = "LONG" if signal == 1 else "SHORT"

    # Log
    log_trade(today, signal, confidence, current_price, action)

    # Print account summary
    try:
        client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
        account = client.get_account()
        log.info(f"\nAccount Summary:")
        log.info(f"Portfolio value: ${float(account.portfolio_value):,.2f}")
        log.info(f"Cash: ${float(account.cash):,.2f}")
        log.info(f"P&L: ${float(account.portfolio_value) - 100000:,.2f}")
    except Exception as e:
        log.warning(f"Could not fetch account: {e}")

    log.info("Done.")

if __name__ == "__main__":
    main()
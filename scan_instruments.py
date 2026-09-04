import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from bar_permute import get_permutation
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Triple Barrier ────────────────────────────────────────────────────────────

def get_daily_vol(close, span0=20):
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0-1], index=close.index[close.shape[0]-df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    return df0.ewm(span=span0).std()

def apply_triple_barrier(close, events, pt_sl):
    out = events[['t1']].copy(deep=True)
    pt = pt_sl[0] * events['trgt'] if pt_sl[0] > 0 else pd.Series(index=events.index, dtype=float)
    sl = -pt_sl[1] * events['trgt'] if pt_sl[1] > 0 else pd.Series(index=events.index, dtype=float)
    for loc, t1 in events['t1'].fillna(close.index[-1]).items():
        df0 = close[loc:t1]
        df0 = (df0 / close[loc] - 1) * events.at[loc, 'side']
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()
    return out

def get_events(close, t_events, pt_sl, trgt, min_ret=0, num_days=5, side=1):
    trgt = trgt.loc[t_events][trgt.loc[t_events] > min_ret]
    t1 = close.index.searchsorted(t_events + pd.Timedelta(days=num_days))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=t_events[:t1.shape[0]])
    events = pd.concat({
        't1': t1,
        'trgt': trgt,
        'side': pd.Series(side, index=trgt.index)
    }, axis=1).dropna(subset=['trgt'])
    df0 = apply_triple_barrier(close, events, pt_sl)
    events['t1'] = df0.dropna(how='all').min(axis=1)
    return events

def get_bins(events, close):
    events_ = events.dropna(subset=['t1'])
    px = close.reindex(events_.index.union(events_['t1'].values).drop_duplicates(), method='bfill')
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index].values - 1
    if 'side' in events_.columns:
        out['ret'] *= events_['side']
    out['bin'] = np.sign(out['ret'])
    out.loc[out['ret'] == 0, 'bin'] = 1
    return out

# ── Features ──────────────────────────────────────────────────────────────────

def build_features(df):
    log_c = np.log(df['close'])
    f = pd.DataFrame(index=df.index)
    f['mom5']       = log_c.diff(5)
    f['mom20']      = log_c.diff(20)
    f['mom60']      = log_c.diff(60)
    f['overnight']  = np.log(df['open'] / df['close'].shift(1))
    f['intraday']   = np.log(df['close'] / df['open'])
    f['vol20']      = log_c.diff().rolling(20).std()
    f['day_of_week']= df.index.dayofweek / 4
    return f.dropna()

feature_cols = ['mom5','mom20','mom60','overnight','intraday','vol20','day_of_week']

# ── Run pipeline on one instrument ───────────────────────────────────────────

def run_pipeline(ticker, n_permutations=300):
    try:
        raw = yf.download(ticker, period='5y', interval='1d', progress=False)
        raw.columns = raw.columns.get_level_values(0)
        raw.columns = [c.lower() for c in raw.columns]
        raw.index = pd.to_datetime(raw.index)
        if raw.index.tz is not None:
            raw.index = raw.index.tz_localize(None)
        raw.index = pd.to_datetime(raw.index.date)
        raw.index.name = None
        df = raw[raw['volume'] > 0].dropna()

        close = df['close']
        daily_vol = get_daily_vol(close)
        events = get_events(close, daily_vol.index, [1,1], daily_vol, min_ret=0.005)
        bins = get_bins(events, close)
        bins.index.name = None

        features = build_features(df)
        features.index.name = None

        common = features.index.intersection(bins.index)
        if len(common) < 50:
            return ticker, None, len(common)

        X = features.loc[common]
        y = (bins.loc[common, 'bin'] > 0).astype(int)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        model.fit(X_scaled, y)

        pred = model.predict(X_scaled)
        signal = pd.Series(np.where(pred > 0, 1, -1), index=common)
        r = np.log(close).diff().shift(-1).reindex(common)
        rets = signal * r
        real_pf = rets[rets > 0].sum() / rets[rets < 0].abs().sum()

        perm_better = 1
        for _ in range(1, n_permutations):
            try:
                perm_df = get_permutation(df)
                perm_df.index = df.index
                perm_df.index.name = None
                perm_close = perm_df['close']
                perm_vol = get_daily_vol(perm_close)
                perm_events = get_events(perm_close, perm_vol.index, [1,1], perm_vol, min_ret=0.005)
                perm_bins = get_bins(perm_events, perm_close)
                perm_bins.index.name = None
                perm_features = build_features(perm_df)
                perm_features.index.name = None
                perm_common = perm_features.index.intersection(perm_bins.index)
                if len(perm_common) < 20:
                    continue
                perm_X = scaler.fit_transform(perm_features.loc[perm_common])
                perm_y = (perm_bins.loc[perm_common, 'bin'] > 0).astype(int)
                perm_model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
                perm_model.fit(perm_X, perm_y)
                perm_pred = perm_model.predict(perm_X)
                perm_signal = pd.Series(np.where(perm_pred > 0, 1, -1), index=perm_common)
                perm_r = np.log(perm_close).diff().shift(-1).reindex(perm_common)
                perm_rets = perm_signal * perm_r
                perm_pf = perm_rets[perm_rets > 0].sum() / perm_rets[perm_rets < 0].abs().sum()
                if perm_pf >= real_pf:
                    perm_better += 1
            except Exception:
                continue

        pval = perm_better / n_permutations
        return ticker, pval, len(common)

    except Exception as e:
        return ticker, None, 0

# ── Scan all instruments ──────────────────────────────────────────────────────

tickers = ['SPY', 'GLD', 'TLT', 'IWM', 'QQQ', 'SLV', 'USO']

print("Scanning instruments for edge...")
print("=" * 50)

results = []
for ticker in tickers:
    print(f"\nTesting {ticker}...")
    t, pval, n = run_pipeline(ticker, n_permutations=300)
    results.append({'ticker': t, 'pval': pval, 'n_samples': n})
    status = f"p={pval:.3f}" if pval is not None else "failed"
    print(f"{ticker}: {status} ({n} samples)")

print("\n" + "=" * 50)
print("RESULTS RANKED BY P-VALUE:")
print("=" * 50)
df_results = pd.DataFrame(results).dropna()
df_results = df_results.sort_values('pval')
print(df_results.to_string(index=False))
print("\nLowest p-value = most promising market")
print("Target: p-value < 0.05 for real edge")
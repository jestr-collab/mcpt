import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from bar_permute import get_permutation

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
    px = close.reindex(
        events_.index.union(events_['t1'].values).drop_duplicates(),
        method='bfill'
    )
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
        f['gld_dxy_ratio']      = np.log(df['close'] / df['dxy'])
        f['gld_dxy_mom10']      = f['gld_dxy_ratio'].diff(10)
        f['gld_dxy_divergence'] = log_c.diff(5) + np.log(df['dxy']).diff(5)
    if 'vix' in df.columns:
        f['vix_level']    = df['vix'] / 100
        f['vix_change']   = df['vix'].pct_change().clip(-5, 5)
        f['vix_above20']  = (df['vix'] > 20).astype(int)
        f['vix_ma_ratio'] = df['vix'] / df['vix'].rolling(20).mean()
        f['vix_spike_25'] = (df['vix'] > 25).astype(int)
        f['vix_spike_30'] = (df['vix'] > 30).astype(int)
        f['vix_5d_change']= df['vix'].pct_change(5).clip(-5, 5)
    if 'tlt' in df.columns:
        f['tlt_mom5']       = np.log(df['tlt']).diff(5)
        f['tlt_mom20']      = np.log(df['tlt']).diff(20)
        f['tlt_change']     = np.log(df['tlt']).diff(1)
        gld_ret             = log_c.diff()
        tlt_ret             = np.log(df['tlt']).diff()
        f['gld_tlt_corr20'] = gld_ret.rolling(20).corr(tlt_ret)
    month = pd.to_datetime(df.index).month
    f['is_september'] = (month == 9).astype(int)
    f['is_november']  = (month == 11).astype(int)
    f['is_january']   = (month == 1).astype(int)
    f['is_june']      = (month == 6).astype(int)
    f['month']        = month / 12
    f['day_of_week']  = df.index.dayofweek / 4
    return f.dropna()

# ── Walk Forward Test ─────────────────────────────────────────────────────────

print("Loading GLD macro data...")
df = pd.read_parquet('GLD_macro.pq')
df.index = pd.to_datetime(df.index)
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)
df.index = pd.to_datetime(df.index.date)
df.index.name = None
df = df[df['volume'] > 0].dropna(subset=['close', 'dxy', 'vix'])
print(f"Total bars: {len(df)}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# Split — train on first 3 years, test on last 2 years
split_date = pd.Timestamp('2025-01-01')
train_df = df[df.index < split_date].copy()
test_df  = df[df.index >= split_date].copy()

print(f"\nTrain: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df)} bars)")
print(f"Test:  {test_df.index[0]} to {test_df.index[-1]} ({len(test_df)} bars)")

# ── Train on training data ────────────────────────────────────────────────────

print("\nTraining on train set...")
train_close = train_df['close'].copy()
train_close.index.name = None

train_vol = get_daily_vol(train_close)
train_vol.index.name = None

train_events = get_events(
    close=train_close,
    t_events=train_vol.index,
    pt_sl=[1, 1],
    trgt=train_vol,
    min_ret=0.003,
    num_days=5,
    side=1
)
train_bins = get_bins(train_events, train_close)
train_bins.index.name = None

train_features = build_features(train_df)
train_features.index.name = None

train_common = train_features.index.intersection(train_bins.index)
X_train = train_features.loc[train_common]
y_train = (train_bins.loc[train_common, 'bin'] > 0).astype(int)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

train_pred = model.predict(X_train_scaled)
train_signal = pd.Series(np.where(train_pred > 0, 1, -1), index=train_common)
train_r = np.log(train_close).diff().shift(-1).reindex(train_common)
train_rets = train_signal * train_r
train_pf = train_rets[train_rets > 0].sum() / train_rets[train_rets < 0].abs().sum()
print(f"Train Profit Factor: {train_pf:.4f}")

# ── Test on unseen data ───────────────────────────────────────────────────────

print("\nTesting on unseen test set...")
test_close = test_df['close'].copy()
test_close.index.name = None

test_vol = get_daily_vol(test_close)
test_vol.index.name = None

test_events = get_events(
    close=test_close,
    t_events=test_vol.index,
    pt_sl=[1, 1],
    trgt=test_vol,
    min_ret=0.003,
    num_days=5,
    side=1
)
test_bins = get_bins(test_events, test_close)
test_bins.index.name = None

test_features = build_features(test_df)
test_features.index.name = None

test_common = test_features.index.intersection(test_bins.index)
X_test = test_features.loc[test_common]
y_test = (test_bins.loc[test_common, 'bin'] > 0).astype(int)

X_test_scaled = scaler.transform(X_test)
test_pred = model.predict(X_test_scaled)
test_signal = pd.Series(np.where(test_pred > 0, 1, -1), index=test_common)
test_r = np.log(test_close).diff().shift(-1).reindex(test_common)
test_rets = test_signal * test_r
test_pf = test_rets[test_rets > 0].sum() / test_rets[test_rets < 0].abs().sum()

print(f"Test Profit Factor: {test_pf:.4f}")
print(f"Test samples: {len(test_common)}")
print(f"Test accuracy: {(test_pred == y_test).mean():.3f}")

# ── Results ───────────────────────────────────────────────────────────────────

print(f"\n{'='*50}")
print("WALK FORWARD RESULTS")
print(f"{'='*50}")
print(f"Train PF: {train_pf:.4f}")
print(f"Test PF:  {test_pf:.4f}")

if test_pf > 1.0:
    print("\nRESULT: Edge holds on unseen data — worth paper trading")
elif test_pf > 0.95:
    print("\nRESULT: Marginal — slight edge on unseen data")
else:
    print("\nRESULT: Edge does not hold on unseen data — overfitting")

# ── Plot cumulative returns ───────────────────────────────────────────────────

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

train_cum = train_rets.cumsum()
test_cum  = test_rets.cumsum()

ax1.plot(train_cum.index, train_cum.values, color='blue')
ax1.set_title(f'Train Cumulative Returns (PF={train_pf:.2f})')
ax1.set_ylabel('Cumulative Log Return')
ax1.axhline(0, color='white', linestyle='--', alpha=0.3)

ax2.plot(test_cum.index, test_cum.values, color='green' if test_pf > 1.0 else 'red')
ax2.set_title(f'Test Cumulative Returns — UNSEEN DATA (PF={test_pf:.2f})')
ax2.set_ylabel('Cumulative Log Return')
ax2.axhline(0, color='white', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('walk_forward_results.png')
plt.show()

print("\nChart saved to walk_forward_results.png")
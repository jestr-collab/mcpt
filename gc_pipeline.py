import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from bar_permute import get_permutation

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
        f['tlt_mom5']       = np.log(df['tlt']).diff(5)
        f['tlt_mom20']      = np.log(df['tlt']).diff(20)
        f['tlt_change']     = np.log(df['tlt']).diff(1)
        gc_ret              = log_c.diff()
        tlt_ret             = np.log(df['tlt']).diff()
        f['gc_tlt_corr20']  = gc_ret.rolling(20).corr(tlt_ret)
    month = pd.to_datetime(df.index).month
    f['is_september'] = (month == 9).astype(int)
    f['is_november']  = (month == 11).astype(int)
    f['is_january']   = (month == 1).astype(int)
    f['is_june']      = (month == 6).astype(int)
    f['month']        = month / 12
    f['day_of_week']  = df.index.dayofweek / 4
    return f.dropna()

# ── Load Data ─────────────────────────────────────────────────────────────────

print("Loading Gold Futures macro data...")
df = pd.read_parquet('GC_macro.pq')
df.index = pd.to_datetime(df.index)
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)
df.index = pd.to_datetime(df.index.date)
df.index.name = None
df = df[df['volume'] > 0].dropna(subset=['close', 'dxy', 'vix'])
print(f"Bars: {len(df)}")

close = df['close'].copy()
close.index.name = None

print("Computing volatility...")
daily_vol = get_daily_vol(close, span0=20)
daily_vol.index.name = None

print("Applying triple barrier labeling...")
events = get_events(close=close, t_events=daily_vol.index,
                    pt_sl=[1,1], trgt=daily_vol, min_ret=0.003, num_days=5, side=1)
bins = get_bins(events, close)
bins.index.name = None
print(f"Labels: {bins['bin'].value_counts().to_dict()}")

features = build_features(df)
features.index.name = None

common = features.index.intersection(bins.index)
X = features.loc[common]
y = (bins.loc[common, 'bin'] > 0).astype(int)
print(f"Samples: {len(common)}, Target: {y.value_counts().to_dict()}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
model.fit(X_scaled, y)

pred = model.predict(X_scaled)
signal = pd.Series(np.where(pred > 0, 1, -1), index=common)
r = np.log(close).diff().shift(-1).reindex(common)
rets = signal * r
real_pf = rets[rets > 0].sum() / rets[rets < 0].abs().sum()
print(f"\nReal Profit Factor: {real_pf:.4f}")

# ── MCPT ──────────────────────────────────────────────────────────────────────

print("\nRunning MCPT on Gold Futures...")
n_permutations = 500
perm_better = 1
permuted_pfs = []

for i in tqdm(range(1, n_permutations)):
    try:
        perm_df = get_permutation(df)
        perm_df.index = df.index
        perm_df.index.name = None
        for col in ['dxy', 'vix', 'tlt']:
            if col in df.columns:
                perm_df[col] = df[col].values
        perm_close = perm_df['close'].copy()
        perm_close.index.name = None
        perm_vol = get_daily_vol(perm_close, span0=20)
        perm_vol.index.name = None
        perm_events = get_events(close=perm_close, t_events=perm_vol.index,
                                 pt_sl=[1,1], trgt=perm_vol, min_ret=0.003, num_days=5, side=1)
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
        permuted_pfs.append(perm_pf)
    except Exception:
        continue

pval = perm_better / n_permutations
print(f"\nP-Value: {pval:.3f}")

if pval < 0.05:
    print("RESULT: Real edge detected in Gold Futures — prop firm ready")
elif pval < 0.10:
    print("RESULT: Marginal edge")
elif pval < 0.20:
    print("RESULT: Promising")
else:
    print("RESULT: No edge yet")

# ── Walk Forward ──────────────────────────────────────────────────────────────

print("\nRunning walk forward test...")
split = pd.Timestamp('2025-01-01')
train_df = df[df.index < split].copy()
test_df  = df[df.index >= split].copy()
print(f"Train: {len(train_df)} bars | Test: {len(test_df)} bars")

train_close = train_df['close'].copy()
train_close.index.name = None
train_vol = get_daily_vol(train_close)
train_vol.index.name = None
train_events = get_events(close=train_close, t_events=train_vol.index,
                          pt_sl=[1,1], trgt=train_vol, min_ret=0.003, num_days=5, side=1)
train_bins = get_bins(train_events, train_close)
train_bins.index.name = None
train_features = build_features(train_df)
train_features.index.name = None
train_common = train_features.index.intersection(train_bins.index)
X_train = train_features.loc[train_common]
y_train = (train_bins.loc[train_common, 'bin'] > 0).astype(int)
scaler2 = StandardScaler()
X_train_scaled = scaler2.fit_transform(X_train)
model2 = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
model2.fit(X_train_scaled, y_train)
train_pred = model2.predict(X_train_scaled)
train_signal = pd.Series(np.where(train_pred > 0, 1, -1), index=train_common)
train_r = np.log(train_close).diff().shift(-1).reindex(train_common)
train_rets = train_signal * train_r
train_pf = train_rets[train_rets > 0].sum() / train_rets[train_rets < 0].abs().sum()

test_close = test_df['close'].copy()
test_close.index.name = None
test_vol = get_daily_vol(test_close)
test_vol.index.name = None
test_events = get_events(close=test_close, t_events=test_vol.index,
                         pt_sl=[1,1], trgt=test_vol, min_ret=0.003, num_days=5, side=1)
test_bins = get_bins(test_events, test_close)
test_bins.index.name = None
test_features = build_features(test_df)
test_features.index.name = None
test_common = test_features.index.intersection(test_bins.index)
X_test = test_features.loc[test_common]
y_test = (test_bins.loc[test_common, 'bin'] > 0).astype(int)
X_test_scaled = scaler2.transform(X_test)
test_pred = model2.predict(X_test_scaled)
test_signal = pd.Series(np.where(test_pred > 0, 1, -1), index=test_common)
test_r = np.log(test_close).diff().shift(-1).reindex(test_common)
test_rets = test_signal * test_r
test_pf = test_rets[test_rets > 0].sum() / test_rets[test_rets < 0].abs().sum()

print(f"\nTrain PF: {train_pf:.4f}")
print(f"Test PF:  {test_pf:.4f}")

if test_pf > 1.0:
    print("WALK FORWARD: Edge holds on unseen data")
else:
    print("WALK FORWARD: Edge does not hold — needs work")

plt.style.use('dark_background')
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
train_rets.cumsum().plot(ax=axes[0], color='blue',
                         title=f'Gold Futures Train (PF={train_pf:.2f})')
test_rets.cumsum().plot(ax=axes[1],
                        color='green' if test_pf > 1.0 else 'red',
                        title=f'Gold Futures Test — UNSEEN (PF={test_pf:.2f})')
for ax in axes:
    ax.axhline(0, color='white', linestyle='--', alpha=0.3)
    ax.set_ylabel('Cumulative Log Return')
plt.tight_layout()
plt.savefig('gc_results.png')
plt.show()
print("\nDone. Chart saved to gc_results.png")
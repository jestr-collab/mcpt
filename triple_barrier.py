import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from bar_permute import get_permutation
from fracdiff import frac_diff_ffd

# ── Triple Barrier Labeling ───────────────────────────────────────────────────

def get_daily_vol(close, span0=20):
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(
        close.index[df0 - 1],
        index=close.index[close.shape[0] - df0.shape[0]:]
    )
    try:
        df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    except Exception as e:
        print(f'Error: {e}')
        return None
    df0 = df0.ewm(span=span0).std()
    return df0

def apply_triple_barrier(close, events, pt_sl):
    out = events[['t1']].copy(deep=True)
    if pt_sl[0] > 0:
        pt = pt_sl[0] * events['trgt']
    else:
        pt = pd.Series(index=events.index, dtype=float)
    if pt_sl[1] > 0:
        sl = -pt_sl[1] * events['trgt']
    else:
        sl = pd.Series(index=events.index, dtype=float)
    for loc, t1 in events['t1'].fillna(close.index[-1]).items():
        df0 = close[loc:t1]
        df0 = (df0 / close[loc] - 1) * events.at[loc, 'side']
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()
    return out

def get_events(close, t_events, pt_sl, trgt, min_ret=0, num_days=1, side=1):
    trgt = trgt.loc[t_events]
    trgt = trgt[trgt > min_ret]
    t1 = close.index.searchsorted(t_events + pd.Timedelta(days=num_days))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=t_events[:t1.shape[0]])
    side_series = pd.Series(side, index=trgt.index)
    events = pd.concat(
        {'t1': t1, 'trgt': trgt, 'side': side_series}, axis=1
    ).dropna(subset=['trgt'])
    df0 = apply_triple_barrier(close, events, pt_sl)
    events['t1'] = df0.dropna(how='all').min(axis=1)
    return events

def get_bins(events, close):
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index].values - 1
    if 'side' in events_.columns:
        out['ret'] *= events_['side']
    out['bin'] = np.sign(out['ret'])
    out.loc[out['ret'] == 0, 'bin'] = 1
    return out

# ── Feature Engineering ───────────────────────────────────────────────────────

def build_features(close_series, open_series, index, d=0.5):
    """Build features using fractional differentiation."""
    log_close = pd.DataFrame({'close': np.log(close_series)})
    log_open  = pd.DataFrame({'open':  np.log(open_series)})
    log_close.index = index
    log_open.index  = index

    frac_close = frac_diff_ffd(log_close, d=d)
    frac_open  = frac_diff_ffd(log_open,  d=d)

    features = pd.DataFrame(index=index)
    features['frac_close']  = frac_close['close']
    features['frac_open']   = frac_open['open']
    features['overnight']   = np.log(open_series / close_series.shift(1))
    features['intraday']    = np.log(close_series / open_series)
    features['vol20']       = np.log(close_series).diff().rolling(20).std()
    features['day_of_week'] = index.dayofweek / 4
    return features.dropna()

feature_cols = ['frac_close', 'frac_open', 'overnight',
                'intraday', 'vol20', 'day_of_week']

# ── Load and Prepare Data ─────────────────────────────────────────────────────

print("Loading SPY data...")
df = pd.read_parquet('SPY_1min.pq')
df.index = pd.to_datetime(df.index)
if df.index.tz is None:
    df.index = df.index.tz_localize('America/New_York')

daily = df.resample('1D').agg({
    'open':   'first',
    'high':   'max',
    'low':    'min',
    'close':  'last',
    'volume': 'sum'
}).dropna()

# Clean index completely
daily.index = daily.index.tz_convert('UTC').tz_localize(None)
daily.index = pd.to_datetime(daily.index.date)
daily.index.name = None
daily = daily[daily['volume'] > 0]  # Remove non-trading days

print(f"Daily bars: {len(daily)}")

close = daily['close'].copy()
close.index.name = None

open_ = daily['open'].copy()
open_.index.name = None

# ── Triple Barrier Labels ─────────────────────────────────────────────────────

print("Computing volatility...")
daily_vol = get_daily_vol(close, span0=20)
daily_vol.index.name = None

print("Applying triple barrier labeling...")
events = get_events(
    close=close,
    t_events=daily_vol.index,
    pt_sl=[1, 1],
    trgt=daily_vol,
    min_ret=0.005,
    num_days=5,
    side=1
)

bins = get_bins(events, close)
bins.index.name = None
print(f"\nLabel distribution:\n{bins['bin'].value_counts()}")
print(f"Total labeled events: {len(bins)}")

# ── Build Features ────────────────────────────────────────────────────────────

print("\nBuilding fractionally differentiated features...")
features = build_features(close, open_, daily.index, d=0.5)
features.index.name = None

print(f"Features shape: {features.shape}")
print(f"Features index sample: {features.index[:3].tolist()}")
print(f"Bins index sample: {bins.index[:3].tolist()}")

# Align
common_idx = features.index.intersection(bins.index)
print(f"Common index size: {len(common_idx)}")

X = features.loc[common_idx]
y = bins.loc[common_idx, 'bin']
y = (y > 0).astype(int)

print(f"Feature matrix shape: {X.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")

if len(X) == 0:
    print("ERROR: No common index found. Check index alignment.")
    print(f"Features dtype: {features.index.dtype}")
    print(f"Bins dtype: {bins.index.dtype}")
    exit()

# ── Train Model ───────────────────────────────────────────────────────────────

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
model.fit(X_scaled, y)

pred = model.predict(X_scaled)
signal = pd.Series(np.where(pred > 0, 1, -1), index=X.index)
r = np.log(close).diff().shift(-1).reindex(X.index)
rets = signal * r
real_pf = rets[rets > 0].sum() / rets[rets < 0].abs().sum()
print(f"\nReal Profit Factor (Triple Barrier + FracDiff): {real_pf:.4f}")

# ── MCPT ──────────────────────────────────────────────────────────────────────

print("\nRunning MCPT...")
n_permutations = 500
perm_better_count = 1
permuted_pfs = []

for i in tqdm(range(1, n_permutations)):
    try:
        perm_df = get_permutation(daily)
        perm_df.index = daily.index
        perm_df.index.name = None

        perm_close = perm_df['close'].copy()
        perm_open  = perm_df['open'].copy()
        perm_close.index.name = None
        perm_open.index.name = None

        perm_vol = get_daily_vol(perm_close, span0=20)
        perm_vol.index.name = None

        perm_events = get_events(
            close=perm_close,
            t_events=perm_vol.index,
            pt_sl=[1, 1],
            trgt=perm_vol,
            min_ret=0.005,
            num_days=5,
            side=1
        )
        perm_bins = get_bins(perm_events, perm_close)
        perm_bins.index.name = None

        perm_features = build_features(perm_close, perm_open, perm_df.index, d=0.5)
        perm_features.index.name = None

        common = perm_features.index.intersection(perm_bins.index)
        if len(common) < 20:
            continue

        perm_X = perm_features.loc[common]
        perm_y = (perm_bins.loc[common, 'bin'] > 0).astype(int)

        perm_X_scaled = scaler.fit_transform(perm_X)
        perm_model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        perm_model.fit(perm_X_scaled, perm_y)

        perm_pred = perm_model.predict(perm_X_scaled)
        perm_signal = pd.Series(np.where(perm_pred > 0, 1, -1), index=common)
        perm_r = np.log(perm_close).diff().shift(-1).reindex(common)
        perm_rets = perm_signal * perm_r
        perm_pf = perm_rets[perm_rets > 0].sum() / perm_rets[perm_rets < 0].abs().sum()

        if perm_pf >= real_pf:
            perm_better_count += 1
        permuted_pfs.append(perm_pf)

    except Exception:
        continue

pval = perm_better_count / n_permutations
print(f"\nP-Value: {pval:.3f}")

if pval < 0.05:
    print("RESULT: Real edge detected")
elif pval < 0.10:
    print("RESULT: Marginal edge — investigate further")
else:
    print("RESULT: No edge yet")

# ── Plot ──────────────────────────────────────────────────────────────────────

plt.style.use('dark_background')
if permuted_pfs:
    pd.Series(permuted_pfs).hist(color='blue', label='Permutations', bins=40)
    plt.axvline(real_pf, color='red',
                label=f'Triple Barrier + FracDiff (PF={real_pf:.2f})')
    plt.xlabel("Profit Factor")
    plt.title(f"Triple Barrier + FracDiff MCPT - P-Value: {pval:.3f}")
    plt.legend()
    plt.savefig('triple_barrier_fracdiff_mcpt.png')
    plt.show()
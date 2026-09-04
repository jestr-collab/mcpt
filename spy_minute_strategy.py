import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from bar_permute import get_permutation
from sklearn.ensemble import RandomForestClassifier

def build_features(df):
    log_c = np.log(df['close'])
    
    df['intrabar'] = np.log(df['close'] / df['open'])
    df['overnight'] = np.log(df['open'] / df['close'].shift(1))
    df['mom5'] = log_c.diff(5)
    df['mom15'] = log_c.diff(15)
    df['mom60'] = log_c.diff(60)
    df['mom390'] = log_c.diff(390)
    df['volatility'] = log_c.diff().rolling(30).std()
    df['target'] = (log_c.diff().shift(-1) > 0).astype(int)
    
    return df

feature_cols = [
    'intrabar', 'overnight', 'mom5', 'mom15',
    'mom60', 'mom390', 'volatility'
]

def train_model(df):
    df = build_features(df.copy()).dropna()
    X = df[feature_cols].to_numpy()
    y = df['target'].to_numpy()
    model = RandomForestClassifier(
        n_estimators=50,
        min_samples_leaf=50,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model

def run_strategy(df, model):
    df = build_features(df.copy()).dropna()
    pred = model.predict(df[feature_cols].to_numpy())
    signal = pd.Series(np.where(pred > 0, 1, -1), index=df.index)
    r = np.log(df['close']).diff().shift(-1)
    rets = signal * r
    pf = rets[rets > 0].sum() / rets[rets < 0].abs().sum()
    return signal, pf

print("Loading data...")
df = pd.read_parquet('SPY_1min.pq')
df.index = pd.to_datetime(df.index)
if df.index.tz is None:
    df.index = df.index.tz_localize('America/New_York')

df = df.between_time('09:30', '16:00')
print(f"Market hours bars: {len(df):,}")

df = df[df.index >= '2026-01-01']
print(f"Using {len(df):,} bars from 2026")

train_df = df.dropna()

print("Training model...")
model = train_model(train_df)
real_signal, real_pf = run_strategy(train_df, model)
print(f"Real Profit Factor: {real_pf:.4f}")

n_permutations = 200
perm_better_count = 1
permuted_pfs = []

print(f"Running MCPT ({n_permutations} permutations)...")
for i in tqdm(range(1, n_permutations)):
    perm_df = get_permutation(train_df)
    perm_model = train_model(perm_df)
    _, perm_pf = run_strategy(perm_df, perm_model)
    if perm_pf >= real_pf:
        perm_better_count += 1
    permuted_pfs.append(perm_pf)

pval = perm_better_count / n_permutations
print(f"\nP-Value: {pval}")

if pval < 0.05:
    print("RESULT: Real edge detected")
elif pval < 0.10:
    print("RESULT: Marginal edge")
else:
    print("RESULT: No edge yet")

plt.style.use('dark_background')
pd.Series(permuted_pfs).hist(color='blue', label='Random Permutations', bins=30)
plt.axvline(real_pf, color='red', label=f'SPY Strategy (PF={real_pf:.2f})')
plt.xlabel("Profit Factor")
plt.title(f"SPY 1-min MCPT - P-Value: {pval:.3f}")
plt.legend()
plt.savefig('spy_minute_mcpt.png')
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from bar_permute import get_permutation
from sklearn.tree import DecisionTreeClassifier

def build_features(df, use_vix=True):
    log_c = np.log(df['close'])
    
    df['mom5'] = log_c.diff(5)
    df['mom20'] = log_c.diff(20)
    df['mom60'] = log_c.diff(60)
    df['overnight'] = np.log(df['open'] / df['close'].shift(1))
    df['intraday'] = np.log(df['close'] / df['open'])
    df['volatility'] = log_c.diff().rolling(20).std()
    df['day_of_week'] = df.index.dayofweek / 4
    
    if use_vix and 'vix' in df.columns:
        df['vix_level'] = df['vix'] / 100
        df['vix_change'] = df['vix'].pct_change()
        df['vix_ma_ratio'] = df['vix'] / df['vix'].rolling(20).mean()
        df['vix_spike'] = (df['vix_change'] > 0.10).astype(int)
        df['vix_above_ma'] = (df['vix'] > df['vix'].rolling(10).mean()).astype(int)
    
    df['target'] = (log_c.diff().shift(-1) > 0).astype(int)
    return df

# Feature sets
price_features = ['mom5', 'mom20', 'mom60', 'overnight', 
                  'intraday', 'volatility', 'day_of_week']
vix_features = ['vix_level', 'vix_change', 'vix_ma_ratio', 
                'vix_spike', 'vix_above_ma']
all_features = price_features + vix_features

def train_model(df, features):
    df = build_features(df.copy(), use_vix='vix' in df.columns).dropna()
    available = [f for f in features if f in df.columns]
    X = df[available].to_numpy()
    y = df['target'].to_numpy()
    model = DecisionTreeClassifier(min_samples_leaf=10, random_state=42)
    model.fit(X, y)
    return model, available

def run_strategy(df, model, feature_cols):
    df = build_features(df.copy(), use_vix='vix' in df.columns).dropna()
    available = [f for f in feature_cols if f in df.columns]
    pred = model.predict(df[available].to_numpy())
    signal = pd.Series(np.where(pred > 0, 1, -1), index=df.index)
    r = np.log(df['close']).diff().shift(-1)
    rets = signal * r
    pf = rets[rets > 0].sum() / rets[rets < 0].abs().sum()
    return signal, pf

# Load data
df = pd.read_parquet('SPY_vix.pq')
df.index = pd.to_datetime(df.index)
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)
train_df = df.dropna(subset=['vix'])

# Real strategy WITH VIX
model, feature_cols = train_model(train_df, all_features)
real_signal, real_pf = run_strategy(train_df, model, feature_cols)
print(f"Candles: {len(train_df)}")
print(f"Real Profit Factor (with VIX): {real_pf:.4f}")

# MCPT using price features only (permuted data has no VIX)
print("Running MCPT on price features only...")
n_permutations = 1000
perm_better_count = 1
permuted_pfs = []

for i in tqdm(range(1, n_permutations)):
    perm_df = get_permutation(train_df)
    perm_model, _ = train_model(perm_df, price_features)
    _, perm_pf = run_strategy(perm_df, perm_model, price_features)
    if perm_pf >= real_pf:
        perm_better_count += 1
    permuted_pfs.append(perm_pf)

pval = perm_better_count / n_permutations
print(f"\nP-Value: {pval}")
if pval < 0.05:
    print("RESULT: VIX adds real edge beyond price alone")
elif pval < 0.10:
    print("RESULT: Marginal edge")
else:
    print("RESULT: No significant edge yet")

plt.style.use('dark_background')
pd.Series(permuted_pfs).hist(color='blue', label='Price-only Permutations', bins=50)
plt.axvline(real_pf, color='red', label=f'SPY+VIX Strategy (PF={real_pf:.2f})')
plt.xlabel("Profit Factor")
plt.title(f"SPY+VIX vs Price-only MCPT - P-Value: {pval:.3f}")
plt.legend()
plt.savefig('spy_vix_mcpt.png')
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from bar_permute import get_permutation
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def build_features(df):
    log_c = np.log(df['close'])
    
    # Price features
    df['mom5'] = log_c.diff(5)
    df['mom20'] = log_c.diff(20)
    df['mom60'] = log_c.diff(60)
    df['volatility'] = log_c.diff().rolling(20).std()
    df['overnight'] = np.log(df['open'] / df['close'].shift(1))
    
    # Google Trends features — the alternative data
    if 'stock market crash' in df.columns:
            df['crash_search'] = df['stock market crash'].pct_change().clip(-5, 5)
            df['recession_search'] = df['recession'].pct_change().clip(-5, 5)
            df['market_search'] = df['stock market'].pct_change().clip(-5, 5)
            df['buy_search'] = df['buy stocks'].pct_change().clip(-5, 5)
            df['fear_index'] = (df['stock market crash'] + df['recession']) / (df['stock market'] + 1)
    # Target
    df['target'] = (log_c.diff().shift(-1) > 0).astype(int)
    
    return df

price_features = ['mom5', 'mom20', 'mom60', 'volatility', 'overnight']
all_features = price_features + [
    'crash_search', 'recession_search', 
    'market_search', 'buy_search', 'fear_index'
]

def train_model(df, features):
    df = build_features(df.copy()).dropna()
    available = [f for f in features if f in df.columns]
    X = df[available].to_numpy()
    y = df['target'].to_numpy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, available

def run_strategy(df, model, scaler, feature_cols):
    df = build_features(df.copy()).dropna()
    available = [f for f in feature_cols if f in df.columns]
    X = scaler.transform(df[available].to_numpy())
    pred = model.predict(X)
    signal = pd.Series(np.where(pred > 0, 1, -1), index=df.index)
    r = np.log(df['close']).diff().shift(-1)
    rets = signal * r
    pf = rets[rets > 0].sum() / rets[rets < 0].abs().sum()
    return signal, pf

# Load
df = pd.read_parquet('SPY_trends.pq')
df.index = pd.to_datetime(df.index)
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)
train_df = df.dropna()

# Real strategy WITH trends
model, scaler, feature_cols = train_model(train_df, all_features)
real_signal, real_pf = run_strategy(train_df, model, scaler, feature_cols)
print(f"Candles: {len(train_df)}")
print(f"Real Profit Factor (with trends): {real_pf:.4f}")

# MCPT — permuted data has no trends so compare against price only
n_permutations = 1000
perm_better_count = 1
permuted_pfs = []

print("Running MCPT...")
for i in tqdm(range(1, n_permutations)):
    perm_df = get_permutation(train_df)
    perm_model, perm_scaler, _ = train_model(perm_df, price_features)
    _, perm_pf = run_strategy(perm_df, perm_model, perm_scaler, price_features)
    if perm_pf >= real_pf:
        perm_better_count += 1
    permuted_pfs.append(perm_pf)

pval = perm_better_count / n_permutations
print(f"\nP-Value: {pval}")

if pval < 0.05:
    print("RESULT: Google Trends adds real edge")
elif pval < 0.10:
    print("RESULT: Marginal edge — worth investigating")
else:
    print("RESULT: No edge yet")

plt.style.use('dark_background')
pd.Series(permuted_pfs).hist(color='blue', label='Price-only Permutations', bins=50)
plt.axvline(real_pf, color='red', label=f'SPY+Trends (PF={real_pf:.2f})')
plt.xlabel("Profit Factor")
plt.title(f"SPY + Google Trends MCPT - P-Value: {pval:.3f}")
plt.legend()
plt.savefig('trends_mcpt.png')
plt.show()
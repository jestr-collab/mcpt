import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from bar_permute import get_permutation
from sklearn.tree import DecisionTreeClassifier

def build_features(df):
    """
    Overnight drift features for ES futures.
    Core idea: close-to-open return predicts next day direction.
    """
    # Overnight return — close yesterday to open today
    df['overnight_ret'] = np.log(df['open'] / df['close'].shift(1))
    
    # Intraday return — open to close
    df['intraday_ret'] = np.log(df['close'] / df['open'])
    
    # 5 day momentum
    df['mom5'] = np.log(df['close']).diff(5)
    
    # 20 day momentum  
    df['mom20'] = np.log(df['close']).diff(20)
    
    # Volatility regime
    df['volatility'] = np.log(df['close']).diff().rolling(20).std()
    
    # Day of week — Monday/Friday effects are real in ES
    df['day_of_week'] = df.index.dayofweek / 4
    
    # Target — does price go up tomorrow?
    df['target'] = (np.log(df['close']).diff().shift(-1) > 0).astype(int)
    
    return df

def train_model(df):
    df = build_features(df).dropna()
    
    feature_cols = ['overnight_ret', 'intraday_ret', 'mom5', 
                    'mom20', 'volatility', 'day_of_week']
    
    X = df[feature_cols].to_numpy()
    y = df['target'].to_numpy()
    
    model = DecisionTreeClassifier(min_samples_leaf=10, random_state=42)
    model.fit(X, y)
    return model, feature_cols

def run_strategy(df, model, feature_cols):
    df = build_features(df).dropna()
    
    pred = model.predict(df[feature_cols].to_numpy())
    signal = pd.Series(np.where(pred > 0, 1, -1), index=df.index)
    
    r = np.log(df['close']).diff().shift(-1)
    rets = signal * r
    if df.index.tz is not None:
        df.index = df.index.tz_convert('UTC').tz_localize(None)
    pf = rets[rets > 0].sum() / rets[rets < 0].abs().sum()
    return signal, pf

# Load data
df = pd.read_parquet('ES_daily.pq')

train_df = df.copy()

# Train and get real profit factor
model, feature_cols = train_model(train_df)
real_signal, real_pf = run_strategy(train_df, model, feature_cols)

print(f"Candles: {len(train_df)}")
print(f"Real Profit Factor: {real_pf:.4f}")

# MCPT
n_permutations = 1000
perm_better_count = 1
permuted_pfs = []

print("Running MCPT on ES overnight drift...")
for i in tqdm(range(1, n_permutations)):
    perm_df = get_permutation(train_df)
    perm_model, _ = train_model(perm_df)
    _, perm_pf = run_strategy(perm_df, perm_model, feature_cols)
    if perm_pf >= real_pf:
        perm_better_count += 1
    permuted_pfs.append(perm_pf)

pval = perm_better_count / n_permutations
print(f"\nP-Value: {pval}")

if pval < 0.05:
    print("RESULT: Real edge detected — worth walk forward testing")
elif pval < 0.10:
    print("RESULT: Marginal edge — test on more data")
else:
    print("RESULT: No edge — iterate features")

plt.style.use('dark_background')
pd.Series(permuted_pfs).hist(color='blue', label='Random Permutations', bins=50)
plt.axvline(real_pf, color='red', label=f'ES Strategy (PF={real_pf:.2f})')
plt.xlabel("Profit Factor")
plt.title(f"ES Overnight Drift MCPT - P-Value: {pval:.3f}")
plt.legend()
plt.savefig('es_mcpt.png')
plt.show()
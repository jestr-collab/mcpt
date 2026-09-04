import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tree_strat_v2 import train_tree_v2, tree_strategy_v2
from bar_permute import get_permutation

df = pd.read_parquet('BTCUSD3600.pq')
df.index = df.index.tz_convert('UTC').tz_localize(None)
df.index = df.index.astype('datetime64[s]')
df['r'] = np.log(df['close']).diff().shift(-1)

train_df = df.dropna()

model, feature_cols = train_tree_v2(train_df)
real_signal, real_pf = tree_strategy_v2(train_df, model, feature_cols)

print(f"Candles: {len(train_df)}")
print(f"Features: {feature_cols}")
print(f"Real Profit Factor: {real_pf:.4f}")

n_permutations = 1000
perm_better_count = 1
permuted_pfs = []

print("Running MCPT v2...")
for perm_i in tqdm(range(1, n_permutations)):
    train_perm = get_permutation(train_df)
    perm_model, _ = train_tree_v2(train_perm)
    _, perm_pf = tree_strategy_v2(train_perm, perm_model, feature_cols)
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
    print("RESULT: No edge yet - keep iterating")

plt.style.use('dark_background')
pd.Series(permuted_pfs).hist(color='blue', label='Random Permutations', bins=50)
plt.axvline(real_pf, color='red', label=f'Your Strategy v2 (PF={real_pf:.2f})')
plt.xlabel("Profit Factor")
plt.title(f"In-sample MCPT v2 - P-Value: {pval:.3f}")
plt.legend()
plt.savefig('mcpt_result_v2.png')
plt.show()
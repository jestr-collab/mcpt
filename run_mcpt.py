import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tree_strat import train_tree, tree_strategy
from bar_permute import get_permutation

df = pd.read_parquet('BTCUSD3600.pq')
df.index = df.index.tz_convert('UTC').tz_localize(None)
df.index = df.index.astype('datetime64[s]')
df['r'] = np.log(df['close']).diff().shift(-1)

# Use all available data now
train_df = df.dropna()

real_tree = train_tree(train_df)
real_is_signal, real_is_pf = tree_strategy(train_df, real_tree)

print(f"Candles: {len(train_df)}")
print(f"Real Profit Factor: {real_is_pf:.4f}")

n_permutations = 1000
perm_better_count = 1
permuted_pfs = []

print("Running MCPT...")
for perm_i in tqdm(range(1, n_permutations)):
    train_perm = get_permutation(train_df)
    perm_nn = train_tree(train_perm)
    _, perm_pf = tree_strategy(train_perm, perm_nn)
    if perm_pf >= real_is_pf:
        perm_better_count += 1
    permuted_pfs.append(perm_pf)

insample_mcpt_pval = perm_better_count / n_permutations
print(f"\nP-Value: {insample_mcpt_pval}")

if insample_mcpt_pval < 0.05:
    print("RESULT: Real edge detected")
elif insample_mcpt_pval < 0.10:
    print("RESULT: Marginal edge - needs more testing")
else:
    print("RESULT: No edge - features need improvement")

plt.style.use('dark_background')
pd.Series(permuted_pfs).hist(color='blue', label='Random Permutations', bins=50)
plt.axvline(real_is_pf, color='red', label=f'Your Strategy (PF={real_is_pf:.2f})')
plt.xlabel("Profit Factor")
plt.title(f"In-sample MCPT - P-Value: {insample_mcpt_pval:.3f}")
plt.legend()
plt.savefig('mcpt_result.png')
plt.show()

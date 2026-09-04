import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree_strat import train_tree, tree_strategy

df = pd.read_parquet('BTCUSD3600.pq')
df.index = df.index.tz_convert('UTC').tz_localize(None)
df.index = df.index.astype('datetime64[s]')
df['r'] = np.log(df['close']).diff().shift(-1)

train_df = df[(df.index.year >= 2024) & (df.index.year < 2025)]

print(f"Training on {len(train_df)} candles")

nn = train_tree(train_df)
is_sig, is_pf = tree_strategy(train_df, nn)
print(f"Profit Factor: {is_pf:.4f}")

(train_df['r'] * is_sig).cumsum().plot()
plt.title('Strategy Cumulative Returns')
plt.savefig('strategy_returns.png')
plt.show()
print("Done")

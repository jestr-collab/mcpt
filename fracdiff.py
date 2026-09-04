import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def get_weights_ffd(d, thres=1e-2):
    w, k = [1.], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def frac_diff_ffd(series, d, thres=1e-2):
    if isinstance(series, pd.Series):
        series = series.to_frame()
    
    w = get_weights_ffd(d, thres)
    width = len(w) - 1
    
    results = {}
    
    for name in series.columns:
        col = series[name].ffill().dropna()
        col_vals = col.values
        col_idx = col.index
        
        out = {}
        for i in range(width, len(col_vals)):
            val = np.dot(w.T, col_vals[i - width:i + 1])[0]
            if np.isfinite(val):
                out[col_idx[i]] = val
        
        results[name] = pd.Series(out)
    
    return pd.concat(results, axis=1)


def find_min_ffd(series, thres=1e-5):
    if isinstance(series, pd.Series):
        series = series.to_frame()
    
    results = {}
    
    for d in np.arange(0, 1.1, 0.1):
        df1 = np.log(series).ffill().dropna()
        df2 = frac_diff_ffd(df1, d=d, thres=thres)
        
        common_idx = df1.index.intersection(df2.index)
        if len(common_idx) < 10:
            continue
        corr = np.corrcoef(
            df1.loc[common_idx, df1.columns[0]].values,
            df2.loc[common_idx].iloc[:, 0].values
        )[0, 1]
        
        df2 = df2.dropna()
        if len(df2) < 20:
            continue
            
        adf = adfuller(df2.iloc[:, 0], maxlag=1, regression='c', autolag=None)
        results[d] = {
            'adfStat': adf[0],
            'pVal': adf[1],
            'corr': corr,
            'stationary': adf[1] < 0.05
        }
    
    return pd.DataFrame(results).T
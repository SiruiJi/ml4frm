import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


def load_raw_data(file_path):
    liq_data = pd.read_csv(file_path)
    liq_data = liq_data[~(liq_data == '#VALUE!').any(axis=1)]
    for col in ['Open', 'ASKHI', 'BIDLO', 'Close', 'PRC', 'Volume', 'RET', 'SHROUT', 'TP', 'vwretx']:
        liq_data[col] = pd.to_numeric(liq_data[col], errors='coerce')

    rolling_window = liq_data.rolling(window=5)
    liq_data['bidlo_min'] = rolling_window['BIDLO'].min()
    liq_data['askhi_max'] = rolling_window['ASKHI'].max()
    liq_data['vol_sum'] = rolling_window['Volume'].sum()
    liq_data['shrout_mean'] = rolling_window['SHROUT'].mean()
    liq_data['prc_mean'] = rolling_window['PRC'].mean()
    return liq_data


def market_based_measures(liq_vol_all):
    liq_vol_all = liq_vol_all.dropna(subset=['vwretx', 'RET', 'Volume'])
    vol_pct_change = liq_vol_all['Volume'].pct_change().dropna()

    X1 = sm.add_constant(liq_vol_all['vwretx'])
    y1 = liq_vol_all['RET']
    ols1 = sm.OLS(y1, X1).fit()
    unsys_resid = ols1.resid

    unsys_resid = unsys_resid.iloc[1:]

    X2 = sm.add_constant(vol_pct_change)
    y2 = unsys_resid ** 2
    ols2 = sm.OLS(y2, X2).fit()
    market_impact = ols2.resid
    market_impact = pd.DataFrame(market_impact)

    return ols2.summary(), market_impact


if __name__ == '__main__':
    file_path = 'D:/PyCharm Community Edition 2023.1.2/Python_Project/Finance/ml4frm/AMD.csv'
    liq_vol_all_ = load_raw_data(file_path)
    market_impact_summary, market_impact_ = market_based_measures(liq_vol_all_)
    print(market_impact_summary)

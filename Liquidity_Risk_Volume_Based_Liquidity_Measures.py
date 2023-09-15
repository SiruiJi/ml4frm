import pandas as pd
import numpy as np
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


def liquidity_ratio(liq_vol_all):

    liq_ratio = []
    for i in range(len(liq_vol_all)):
        liq_ratio.append(liq_vol_all['PRC'][i+1:i+6].sum() * liq_vol_all['Volume'][i+1:i+6].sum() /
                         (np.abs(liq_vol_all['PRC'][i+1:i+6].mean() - liq_vol_all['PRC'][i:i+5].mean())))
    liq_ratio = pd.DataFrame(liq_ratio)

    return liq_ratio


def hui_heubel_ratio(liq_vol_all):
    lhh = []
    for i in range(len(liq_vol_all)):
        lhh.append((liq_vol_all['PRC'][i:i+5].max() - liq_vol_all['PRC'][i:i+5].min()) /
                   liq_vol_all['PRC'][i:i+5].min() /
                   (liq_vol_all['Volume'][i:i+5].sum() /
                    liq_vol_all['SHROUT'][i:i+5].mean() *
                    liq_vol_all['PRC'][i:i+5].mean()))
    lhh = pd.DataFrame(lhh)
    return lhh


def turn_over_ratio(liq_vol_all):
    turnover_ratio = []
    for i in range(len(liq_vol_all)):
        turnover_ratio.append((1/liq_vol_all['Volume'].count()) *
                              (liq_vol_all['Volume'][i:i+1].sum() /
                               liq_vol_all['SHROUT'][i:i+1].sum()))
    turnover_ratio = pd.DataFrame(turnover_ratio)
    return turnover_ratio


if __name__ == '__main__':
    file_path = 'D:/PyCharm Community Edition 2023.1.2/Python_Project/Finance/ml4frm/AMD.csv'
    liq_vol_all_ = load_raw_data(file_path)
    liq_ratio_ = liquidity_ratio(liq_vol_all_)
    lhh_ = hui_heubel_ratio(liq_vol_all_)
    turnover_ratio_ = turn_over_ratio(liq_vol_all_)
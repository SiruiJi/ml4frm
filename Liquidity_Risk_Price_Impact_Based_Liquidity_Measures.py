import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_raw_data(file_path):
    liq_data = pd.read_csv(file_path)
    liq_data = liq_data[~(liq_data == '#VALUE!').any(axis=1)]
    liq_data['RET'] = pd.to_numeric(liq_data['RET'])

    rolling_five = []
    for i in range(len(liq_data)):
        rolling_five.append(liq_data[i:i+5].agg({'BIDLO': 'min',
                                                 'ASKHI': 'max',
                                                 'Volume': 'sum',
                                                 'SHROUT': 'mean',
                                                 'PRC': 'mean'}))
    rolling_five_df = pd.DataFrame(rolling_five)
    rolling_five_df.columns = ['bidlo_min', 'askhi_max', 'vol_sum', 'shrout_mean', 'prc_mean']
    liq_vol_all = pd.concat([liq_data, rolling_five_df], axis=1)
    return liq_vol_all


def amihud_illiquidity(liq_vol_all):
    dvol = []
    for i in range(len(liq_vol_all)):
        dvol.append((liq_vol_all['PRC'][i:i+5] * liq_vol_all['Volume'][i:i+5]).sum())
    dvol = pd.DataFrame(dvol)

    amihud = []
    for i in range(len(liq_vol_all)):
        amihud.append((1 / liq_vol_all['RET'].count()) *
                      (np.sum(np.abs(liq_vol_all['RET'][i:i+1])) /
                       np.sum(dvol[i:i+1])))
    amihud = pd.DataFrame(amihud)

    return amihud


def return_to_turnover(liq_vol_all):
    turnover_ratio = []
    for i in range(len(liq_vol_all)):
        turnover_ratio.append((1 / liq_vol_all['Volume'].count()) *
                              (liq_vol_all['Volume'][i:i + 1].sum() /
                               liq_vol_all['SHROUT'][i:i + 1].sum()))
    florackis = []
    for i in range(len(liq_vol_all)):
        florackis.append((1 / liq_vol_all['RET'].count()) *
                         (np.sum(np.abs(liq_vol_all['RET'][i:i+1]) /
                                 turnover_ratio[i:i+1])))
    florackis = pd.DataFrame(florackis)

    return florackis


def coefficient_of_elasticity(liq_vol_all):

    vol_diff_pct = liq_vol_all['Volume'].diff().pct_change()

    price_diff_pct = liq_vol_all['PRC'].diff().pct_change()

    cet = []
    for i in range(len(liq_vol_all)):
        cet.append(np.sum(vol_diff_pct[i:i+1]) /
                   np.sum(price_diff_pct[i:i+1]))
    cet = pd.DataFrame(cet)
    return cet


if __name__ == '__main__':
    file_path = 'D:/PyCharm Community Edition 2023.1.2/Python_Project/Finance/ml4frm/AMD.csv'
    liq_vol_all_ = load_raw_data(file_path)
    amihud_ = amihud_illiquidity(liq_vol_all_)
    florackis_ = return_to_turnover(liq_vol_all_)
    cet_ = coefficient_of_elasticity(liq_vol_all_)
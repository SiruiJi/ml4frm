import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_raw_data(file_path):
    liq_data = pd.read_csv(file_path)
    liq_data = liq_data[~(liq_data == '#VALUE!').any(axis=1)]

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


def percentage_quoted_and_effective_bid_ask_spread(liq_vol_all):

    mid_price = (liq_vol_all.ASKHI + liq_vol_all.BIDLO) / 2

    percent_quote_ba = (liq_vol_all.ASKHI - liq_vol_all.BIDLO) / mid_price
    percent_effective_ba = 2 * abs((liq_vol_all.PRC - mid_price)) / mid_price

    return percent_quote_ba, percent_effective_ba


def rolls_spread_estimate(liq_vol_all):
    price_diff = liq_vol_all.PRC.diff()
    price_diff.dropna(inplace=True)

    roll = []
    for i in range(len(price_diff)-5):
        roll_cov = np.cov(price_diff[i:i+5], price_diff[i+1:i+6])
        if roll_cov[0, 1] < 0:
            roll.append(2 * np.sqrt(-roll_cov[0, 1]))
        else:
            roll.append(2 * np.sqrt(np.abs(roll_cov[0, 1])))
    roll = pd.DataFrame(roll)

    return roll


def corwin_schultz_spread(liq_vol_all):
    global beta_array, gamma_array
    gamma = []
    for i in range(len(liq_vol_all)-1):
        gamma.append((max(liq_vol_all['ASKHI'].iloc[i+1],
                          liq_vol_all['ASKHI'].iloc[i]) -
                      min(liq_vol_all['BIDLO'].iloc[i+1],
                          liq_vol_all['BIDLO'].iloc[i])) ** 2)
        gamma_array = np.array(gamma)

    beta = []
    for i in range(len(liq_vol_all)-1):
        beta.append((liq_vol_all['ASKHI'].iloc[i+1] - liq_vol_all['BIDLO'].iloc[i+1]) ** 2 +
                    (liq_vol_all['ASKHI'].iloc[i] - liq_vol_all['BIDLO'].iloc[i]) ** 2)
        beta_array = np.array(beta)

    alpha = ((np.sqrt(2 * beta_array) - np.sqrt(beta_array)) / (3 - (2 * np.sqrt(2)))) - np.sqrt(gamma_array / (3 - (2 * np.sqrt(2))))
    cs_spread = (2 * np.exp(alpha - 1)) / (1 + np.exp(alpha))
    cs_spread = pd.DataFrame(cs_spread)
    return cs_spread


if __name__ == '__main__':
    file_path = 'D:/PyCharm Community Edition 2023.1.2/Python_Project/Finance/ml4frm/AMD.csv'
    liq_vol_all_ = load_raw_data(file_path)
    percent_quote_ba_, percent_effective_ba_ = percentage_quoted_and_effective_bid_ask_spread(liq_vol_all_)
    roll_ = rolls_spread_estimate(liq_vol_all_)
    cs_spread_ = corwin_schultz_spread(liq_vol_all_)

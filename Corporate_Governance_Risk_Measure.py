import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from sklearn.covariance import EllipticEnvelope


def load_raw_data(ticker, start_date, end_date):
    crash_data = pd.DataFrame()
    for i in ticker:
        raw_data = yf.download(i, start_date, end_date)
        crash_df = pd.DataFrame()
        crash_df['RET'] = (raw_data['Adj Close'] / raw_data['Adj Close'].shift(1)) - 1
        crash_df.index = raw_data.index
        crash_df['BIDLO'] = raw_data['Low']
        crash_df['ASKHI'] = raw_data['High']
        crash_df['PRC'] = raw_data['Close']
        crash_df['VOL'] = raw_data['Volume']
        typical_price = (raw_data['High'] + raw_data['Low'] + raw_data['Close']) / 3
        crash_df['VWAP'] = (typical_price * raw_data['Volume']).cumsum() / raw_data['Volume'].cumsum()
        crash_df['vwretx'] = (crash_df['VWAP'] / crash_df['VWAP'].shift(1)) - 1
        crash_df['TICKER'] = i
        crash_df.dropna(inplace=True)
        crash_data = pd.concat([crash_data, crash_df])

    return crash_data


def weekly_hist_gram(crash_data):
    crash_dataw = crash_data.groupby('TICKER').resample('W').agg({'RET': 'mean', 'vwretx': 'mean', 'VOL': 'mean',
                                                                  'BIDLO': 'mean', 'ASKHI': 'mean', 'PRC': 'mean'})
    crash_dataw = crash_dataw.reset_index()
    crash_dataw.dropna(inplace=True)
    stocks = crash_data.TICKER.unique()
    plt.figure(figsize=(12, 8))
    k = 1
    for i in stocks[:4]:
        plt.subplot(2, 2, k)
        plt.hist(crash_dataw[crash_dataw.TICKER == i]['RET'])
        plt.title('Histogram of ' + i)
        k += 1
    plt.show()
    return crash_dataw, stocks


def firm_specific_weekly_return(crash_dataw, stocks):
    residuals_dict = {}  # We will store residuals for each stock in this dictionary

    for i in stocks:
        Y = crash_dataw.loc[crash_dataw['TICKER'] == i]['RET'].values
        X = crash_dataw.loc[crash_dataw['TICKER'] == i]['vwretx'].values
        X = sm.add_constant(X)

        X_transformed = X[2:-2] + X[1:-3] + X[0:-4] + X[3:-1] + X[4:]
        ols = sm.OLS(Y[2:-2], X_transformed).fit()

        residuals_stock = ols.resid
        residuals_dict[i] = list(map(lambda x: np.log(1 + x), residuals_stock))

    crash_data_sliced = pd.DataFrame([])
    for i in stocks:
        crash_data_sliced = pd.concat([crash_data_sliced, crash_dataw.loc[crash_dataw.TICKER == i][2:-2]],
                                      ignore_index=True)
    print(crash_data_sliced.head())

    envelope = EllipticEnvelope(contamination=0.02, support_fraction=1)
    ee_predictions = {}

    for i in stocks:
        stock_residuals = np.array(residuals_dict[i]).reshape(-1, 1)
        if stock_residuals.shape[0] < 2:
            print(f"Skipping stock {i} due to insufficient residuals.")
            continue  # Skip the current iteration and move to the next stock
        envelope.fit(stock_residuals)
        ee_predictions[i] = envelope.predict(stock_residuals)

    transform = []
    for i in stocks:
        if i in ee_predictions:  # Ensure we only process stocks that were not skipped
            for j in range(len(ee_predictions[i])):
                transform.append(np.where(ee_predictions[i][j] == 1, 0, -1))

    crash_data_sliced = crash_data_sliced.reset_index()
    crash_data_sliced['residuals'] = np.concatenate(list(residuals_dict.values()))
    crash_data_sliced['neg_outliers'] = np.where((np.array(transform)) == -1, 1, 0)
    crash_data_sliced.loc[(crash_data_sliced.neg_outliers == 1) & (crash_data_sliced.residuals > 0), 'neg_outliers'] = 0

    plt.figure(figsize=(12, 8))
    k = 1

    for i in stocks[8:12]:
        plt.subplot(2, 2, k)
        crash_data_sliced['residuals'][crash_data_sliced.TICKER == i].hist(label='normal', bins=30, color='gray')
        outliers = crash_data_sliced['residuals'][
            (crash_data_sliced.TICKER == i) & (crash_data_sliced.neg_outliers > 0)]
        outliers.hist(color='black', label='anomaly')
        plt.title(i)
        plt.legend()
        k += 1
    plt.show()

    return crash_data_sliced


def weekly_to_annual_data(crash_data_sliced, crash_data, crash_dataw):
    crash_data_sliced = crash_data_sliced.set_index('Date')
    crash_data_sliced.index = pd.to_datetime(crash_data_sliced.index)

    std = crash_data.groupby('TICKER')['RET'].resample('W').std().reset_index()
    crash_dataw['std'] = pd.DataFrame(std['RET'])

    yearly_data = crash_data_sliced.groupby('TICKER').resample('Y')['residuals'].agg(['mean', 'std']).reset_index()
    print(yearly_data.head())

    merge_crash = pd.merge(crash_data_sliced.reset_index(), yearly_data, how='outer', on=['TICKER', 'Date'])
    merge_crash[['annual_mean', 'annual_std']] = merge_crash.sort_values(by=['TICKER', 'Date']).iloc[:, -2:].fillna(
        method='bfill')
    merge_crash['residuals'] = merge_crash.sort_values(by=['TICKER', 'Date'])['residuals'].fillna(method='ffill')
    merge_crash = merge_crash.drop(merge_crash.iloc[:, -4:-2], axis=1)

    return merge_crash


def crash_risk_measure(merge_crash, stocks):
    crash_risk_out = []
    for j in stocks:
        for k in range(len(merge_crash[merge_crash.TICKER == j])):
            if merge_crash[merge_crash.TICKER == j]['residuals'].iloc[k] < \
                    merge_crash[merge_crash.TICKER == j]['annual_mean'].iloc[k] - \
                    3.09 * merge_crash[merge_crash.TICKER == j]['annual_std'].iloc[k]:
                crash_risk_out.append(1)
            else:
                crash_risk_out.append(0)
    merge_crash['crash_risk'] = crash_risk_out
    print(merge_crash['crash_risk'].value_counts())

    merge_crash = merge_crash.set_index('Date')
    merge_crash_annual = merge_crash.groupby('TICKER').resample('1Y')['crash_risk'].sum().reset_index()

    down = []
    for j in range(len(merge_crash)):
        if merge_crash['residuals'].iloc[j] < merge_crash['annual_mean'].iloc[j]:
            down.append(1)
        else:
            down.append(0)

    merge_crash = merge_crash.reset_index()
    merge_crash['down'] = pd.DataFrame(down)
    merge_crash['up'] = 1 - merge_crash['down']
    down_residuals = merge_crash[merge_crash.down == 1][['residuals', 'TICKER', 'Date']]
    up_residuals = merge_crash[merge_crash.up == 1][['residuals', 'TICKER', 'Date']]

    down_residuals['residuals_down_sq'] = down_residuals['residuals'] ** 2
    down_residuals['residuals_down_cubic'] = down_residuals['residuals'] ** 3
    up_residuals['residuals_up_sq'] = up_residuals['residuals'] ** 2
    up_residuals['residuals_up_cubic'] = up_residuals['residuals'] ** 3
    down_residuals['down_residuals'] = down_residuals['residuals']
    up_residuals['up_residuals'] = up_residuals['residuals']
    del down_residuals['residuals']
    del up_residuals['residuals']
    merge_crash['residuals_sq'] = merge_crash['residuals'] ** 2
    merge_crash['residuals_cubic'] = merge_crash['residuals'] ** 3

    merge_crash_all = merge_crash.merge(down_residuals, on=['TICKER', 'Date'], how='outer')
    merge_crash_all = merge_crash_all.merge(up_residuals, on=['TICKER', 'Date'], how='outer')
    cols = ['BIDLO', 'ASKHI', 'residuals', 'annual_std', 'residuals_sq', 'residuals_cubic', 'down', 'up',
            'residuals_up_sq', 'residuals_down_sq', 'neg_outliers']
    merge_crash_all = merge_crash_all.set_index('Date')
    merge_grouped = merge_crash_all.groupby('TICKER')[cols].resample('1Y').sum().reset_index()
    merge_grouped['neg_outliers'] = np.where(merge_grouped.neg_outliers >= 1, 1, 0)

    merge_grouped = merge_grouped.set_index('Date')
    merge_all = merge_grouped.groupby('TICKER').resample('1Y').agg({'down': ['sum', 'count'],
                                                                    'up': ['sum', 'count']}).reset_index()
    print(merge_all.head())

    merge_grouped['down'] = merge_all['down']['sum'].values
    merge_grouped['up'] = merge_all['up']['sum'].values
    merge_grouped['count'] = merge_grouped['down'] + merge_grouped['up']

    merge_grouped = merge_grouped.reset_index()
    merge_grouped['duvol'] = np.log(((merge_grouped['up'] - 1) * merge_grouped['residuals_down_sq']) /
                                    ((merge_grouped['down'] - 1) * merge_grouped['residuals_up_sq']))
    print(merge_grouped.groupby('TICKER')['duvol'].mean())

    merge_grouped['ncskew'] = - (((merge_grouped['count'] * (merge_grouped['count'] - 1) ** (3 / 2)) *
                                  merge_grouped['residuals_cubic']) / (((merge_grouped['count'] - 1) *
                                (merge_grouped['count'] - 2)) * merge_grouped['residuals_sq'] ** (3 / 2)))
    print(merge_grouped.groupby('TICKER')['ncskew'].mean())

    merge_grouped['crash_risk'] = merge_crash_annual['crash_risk']
    merge_grouped['crash_risk'] = np.where(merge_grouped.crash_risk >= 1, 1, 0)
    merge_crash_all_grouped2 = merge_crash_all.groupby('TICKER')[['VOL', 'PRC']].resample('1Y').mean().reset_index()
    merge_grouped[['VOL', 'PRC']] = merge_crash_all_grouped2[['VOL', 'PRC']]
    print(merge_grouped[['ncskew', 'duvol']].corr())
    return merge_grouped


if __name__ == '__main__':
    ticker = ['ABBV', 'GOOGL', 'JNJ', 'DLTR', 'HLT', 'JPM', 'DEO', 'PG', 'ALB', 'BA', 'NVDA', 'LUV', 'PEP', 'TSM',
              'SPY', '^VIX', 'GLD']
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2023, 1, 1)
    crash_data_ = load_raw_data(ticker, start_date, end_date)
    crash_dataw_, stocks_ = weekly_hist_gram(crash_data_)
    crash_data_sliced_ = firm_specific_weekly_return(crash_dataw_, stocks_)
    merge_crash_ = weekly_to_annual_data(crash_data_sliced_, crash_data_, crash_dataw_)
    merge_grouped_ = crash_risk_measure(merge_crash_, stocks_)

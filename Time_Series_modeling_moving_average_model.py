import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import numpy as np
import datetime as dt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm


def load_raw_data(tickers, start_date, end_date):
    price = yf.download(tickers, start_date, end_date)['Adj Close']
    price = price.reindex(columns=tickers)
    ret = np.log(price / price.shift(1))
    ret.dropna(inplace=True)
    return ret


def stationary_test(tickers, ret):
    for i in tickers:
        stat_test = adfuller(ret[i])[0:2]
        print("The ADF test statistic and p-value of {} are {}".format(i, stat_test))


def train_test_split(tickers, ret):
    split = int(len(ret['SPY'].values) * 0.95)

    train_data = pd.DataFrame()
    for i in tickers:
        train_data[i] = ret[i].iloc[:split]

    test_data = pd.DataFrame()
    for i in tickers:
        test_data[i] = ret[i].iloc[split:]

    return train_data, test_data


def lag_determine_by_acf(train_data, tickers):
    fig, ax = plt.subplots(len(tickers), 1, figsize=(10, 6 * len(tickers)))  # create a subplot for each ticker
    plt.tight_layout()
    for idx, ticker in enumerate(tickers):
        sm.graphics.tsa.plot_acf(train_data[ticker], lags=30, ax=ax[idx], title=ticker + ' acf')

    plt.show()


def moving_average_model(train_data, start_date, end_date, tickers):
    for i in tickers:
        short_moving_average = train_data[i].rolling(window=9).mean()
        long_moving_average = train_data[i].rolling(window=22).mean()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_data[i].loc[start_date: end_date].index,
                train_data[i].loc[start_date: end_date],
                label=i + ' Stock return', linestyle='--')
        ax.plot(short_moving_average.loc[start_date: end_date].index,
                short_moving_average.loc[start_date: end_date],
                label='Short MA', linestyle='solid')
        ax.plot(long_moving_average.loc[start_date: end_date].index,
                long_moving_average.loc[start_date: end_date],
                label='Long MA', linestyle='solid')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    tickers = ['SPY', 'NVDA']
    start_date = dt.datetime(2020, 1, 1)
    end_date = dt.datetime(2023, 6, 18)
    ret_ = load_raw_data(tickers, start_date, end_date)
    stationary_test(tickers, ret_)
    train_data_, test_data_ = train_test_split(tickers, ret_)
    lag_determine_by_acf(train_data_, tickers)
    moving_average_model(train_data_, start_date, end_date, tickers)

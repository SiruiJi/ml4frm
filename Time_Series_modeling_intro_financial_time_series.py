import pandas as pd
import yfinance as yf
import datetime as dt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
'''Time series has four components: trend, seasonality, cyclicality, and residual'''
def time_series_components(ticker, start_date, end_date):
    price = yf.download(ticker, start_date, end_date, interval='1mo')['Adj Close']
    # ret = np.log(price / price.shift(1))
    # ret.dropna(inplace=True)  # price = price.dropna()
    seasonal_decompose(price, period=12).plot()
    plt.show()
    return price
'''noise and the cyclical cimponent are put together under the residual component'''

def auto_correlation(price):
    sm.graphics.tsa.plot_acf(price, lags=30)
    plt.xlabel('Number of Lags')
    plt.show()


def partial_auto_correlation(price):
    sm.graphics.tsa.plot_pacf(price, lags=15)
    plt.xlabel('Number of Lags')
    plt.show()


def stationary_test(price):
    stat_test = adfuller(price)[0:2]
    print("The test statistic and p-value of ADF test are {}".format(stat_test))


if __name__ == '__main__':
    ticker = '^GSPC'
    start_date = dt.datetime(2020, 1, 1)
    end_date = dt.datetime(2023, 6, 1)
    price_ = time_series_components(ticker, start_date, end_date)
    auto_correlation(price_)
    partial_auto_correlation(price_)
    stationary_test(price_)

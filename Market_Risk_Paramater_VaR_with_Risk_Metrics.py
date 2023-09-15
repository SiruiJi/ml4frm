import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def load_raw_data(tickers, start_date, end_date):
    price = yf.download(tickers, start_date, end_date, interval='1d')['Adj Close']
    price = pd.DataFrame(price)
    price = price.reindex(columns=tickers)
    return price


def risk_metrics(price):
    ret = np.log(price / price.shift(1))
    ret = ret.dropna()
    sigma2_df = pd.DataFrame(index=ret.index, columns=tickers)
    for ticker in tickers:
        ret_values = ret[ticker].values
        T = len(ret_values)
        sigma2 = np.zeros(T)
        sigma2[0] = 0.014 ** 2
        for i in range(1, T):
            sigma2[i] = 0.94 * sigma2[i - 1] + 0.06 * ret_values[i - 1] ** 2
        sigma2_df[ticker] = sigma2

    return ret, sigma2_df


def parameter_var(ret, sigma2_df):
    var_df = pd.DataFrame(index=sigma2_df.index, columns=tickers)
    for ticker in tickers:
        mean_returns = ret[ticker].mean()
        for time_point in sigma2_df.index:
            sigma = np.sqrt(sigma2_df.loc[time_point, ticker])
            var = norm.ppf(0.01, loc=mean_returns, scale=sigma)
            var_df.loc[time_point, ticker] = -var

    for ticker in var_df.columns:
        plt.plot(var_df.index, var_df[ticker], label=ticker)
    plt.xlabel('Date')
    plt.ylabel('VaR')
    plt.title('Value at Risk for each ticker')
    plt.legend()
    plt.show()

    return var_df


if __name__ == '__main__':
    tickers = ['NWSA', 'FOXA', 'CMCSA', '^GSPC', 'LYV', 'NKE', 'NFLX', 'SONY', 'ATVI', 'CHTR', 'MAR', 'LULU', 'LVS', 'DKNG', 'DIS']
    start_date = dt.datetime(2020, 1, 1)
    end_date = dt.datetime(2023, 7, 31)
    price_ = load_raw_data(tickers, start_date, end_date)
    ret_, sigma2_df_ = risk_metrics(price_)
    var_df_ = parameter_var(ret_, sigma2_df_)


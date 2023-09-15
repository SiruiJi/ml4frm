import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from arch import arch_model
from scipy.stats import norm
import matplotlib.pyplot as plt


def load_raw_data(ticker, start_date, end_date):
    price = yf.download(ticker, start_date, end_date)['Adj Close']
    ret =100 * price.pct_change()[1:]
    ret.dropna(inplace=True)
    return ret


def find_best_paras(ret):
    global garch, best_param, q
    bic_garch = []
    for p in range(1, 5):
        for q in range(1, 5):
            garch = arch_model(ret, mean='zero', vol='GARCH', p=p, o=0, q=q).fit(disp='off')
        bic_garch.append(garch.bic)
        if garch.bic == np.min(bic_garch):
            best_param = p, q
    garch = arch_model(ret, mean='zero', vol='GARCH', p=best_param[0], o=0, q=best_param[1]).fit(disp='off')
    print(garch.summary())


def var_calculation(ret):
    var_fhs = np.empty_like(ret)
    for i in range(500, len(ret)):
        # Get window of past 500 days' returns
        ret_window = ret.iloc[i - 500:i]

        garch = arch_model(ret_window, vol='Garch', p=1, q=1).fit(disp='off')
        omega, alpha, beta = garch.params['omega'], garch.params['alpha[1]'], garch.params['beta[1]']
        sigma = np.sqrt(omega + alpha * ret_window.iloc[-1] ** 2 + beta * garch.conditional_volatility[-1] ** 2)

        # Filtered Historical Simulation
        standardized_returns = ret_window / garch.conditional_volatility
        var_fhs[i] = -sigma * np.percentile(standardized_returns, 1)

    plt.plot(var_fhs, label='FHS')
    plt.show()
    return var_fhs


def next_day_var(ret):
    # Train the GARCH model on the entire dataset
    #garch = arch_model(ret, vol='Garch', p=1, q=1).fit(disp='off')

    garch = arch_model(ret.iloc[-500:], vol='Garch', p=1, q=1).fit(disp='off')

    # Predict the volatility for the next day
    forecasts = garch.forecast(start=0)
    next_day_volatility = np.sqrt(forecasts.variance.iloc[-1, 0])

    # Standardize the returns of the last 500 days
    standardized_returns = ret.iloc[-500:] / garch.conditional_volatility[-500:]

    # Estimate the VaR for the next day
    next_day_var = -next_day_volatility * np.percentile(standardized_returns, 1)

    print(f"Next day VaR: {next_day_var}")


if __name__ == '__main__':
    ticker = '^GSPC'
    start_date = dt.datetime(2016, 1, 1)
    end_date = dt.datetime(2023, 8, 1)
    ret_ = load_raw_data(ticker, start_date, end_date)
    # find_best_paras(ret_)
    var_fhs = var_calculation(ret_)
    next_day_var(ret_)

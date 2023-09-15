import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from arch import arch_model
from scipy.stats import norm
import matplotlib.pyplot as plt


def load_raw_data(ticker, start_date, end_date):
    price = yf.download(ticker, start_date, end_date)['Adj Close']
    ret = 100 * price.pct_change()[1:]
    ret.dropna(inplace=True)
    return ret


def find_best_params(ret):
    global garch, q, best_param
    bic_garch = []
    for p in range(1, 5):
        for q in range(1, 5):
            garch = arch_model(ret, mean='zero', vol='GARCH', p=p, o=0, q=q).fit(disp='off')
        bic_garch.append(garch.bic)
        if garch.bic == np.min(bic_garch):
            best_param = p, q
    garch = arch_model(ret, mean='zero', vol='GARCH', p=best_param[0], o=0, q=best_param[1]).fit(disp='off')
    print(garch.summary())


def fit_garch_model(ret):
    garch = arch_model(ret, vol='GARCH', p=1, q=1).fit(disp='off')
    omega, alpha, beta = garch.params[1:4]
    sigma = garch.conditional_volatility
    ret_stand = ret / sigma

    # Predict the next volatility
    sigma_pred = np.sqrt(omega + alpha * ret[-1] ** 2 + beta * sigma[-1] ** 2)
    long_run_vol = np.sqrt(omega / (1 - alpha - beta))
    return ret_stand, sigma_pred, omega, alpha, beta


def monte_carlo_simulation(ret_stand, omega, alpha, beta, sigma_pred, MC, T):
    shock = np.random.choice(ret_stand, (MC, T))
    ReturnMC = np.empty((MC, T))
    for i in range(MC):
        sigmapredMC = sigma_pred
        for j in range(T):
            ReturnMC[i, j] = sigmapredMC * shock[i, j]
            sigmapredMC = np.sqrt(omega + alpha * ReturnMC[i, j] ** 2 + beta * sigmapredMC ** 2)
    ReturnMCT = np.cumsum(ReturnMC, axis=1)

    # Compute Value at Risk (VaR)
    VaRMC = np.percentile(-ReturnMCT, 1, axis=0)

    # Compute Expected Shortfall (ES)
    ESMC = [np.mean(-ReturnMCT[ReturnMCT[:, i] <= -VaRMC[i], i]) for i in range(T)]

    # Plot VaR
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(VaRMC, lw=4)
    plt.title('VaR')
    plt.subplot(1, 2, 2)
    plt.plot(VaRMC / np.sqrt(np.arange(1, T + 1)) / VaRMC[0], lw=4)
    plt.title('Scaled VaR')
    plt.show()

    # Plot ES
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(ESMC, lw=4)
    plt.title('ES')
    plt.subplot(1, 2, 2)
    plt.plot(ESMC / np.sqrt(np.arange(1, T + 1)) / ESMC[0], lw=4)
    plt.title('Scaled ES')
    plt.show()


if __name__ == '__main__':
    ticker = '^GSPC'
    start_date = dt.datetime(2020, 1, 1)
    end_date = dt.datetime(2023, 8, 8)
    ret_ = load_raw_data(ticker, start_date, end_date)
    find_best_params(ret_)
    ret_stand_, sigma_pred_, omega_, alpha_, beta_ = fit_garch_model(ret_)
    monte_carlo_simulation(ret_stand_, omega_, alpha_, beta_, sigma_pred_, MC=10000, T=500)


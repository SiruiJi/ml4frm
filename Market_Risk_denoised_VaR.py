import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np
from scipy.stats import norm
from pypfopt import risk_models


def load_raw_data(tickers, start_date, end_date):
    price = yf.download(tickers, start_date, end_date)['Adj Close']
    price = pd.DataFrame(price)
    price = price.reindex(columns=tickers)
    return price


def de_noised_data_info(price):
    returns = np.log(price / price.shift(1)).dropna()

    returns_mean = returns.mean()
    returns_std = returns.std()
    weights = np.random.random(len(returns.columns))
    '''np.random.random(size) generates random floats in the half-open interval [0.0, 1.0). The argument inside the 
    parentheses len(returns.columns) is the number of stocks in your portfolio, i.e., the number of columns in your 
    returns DataFrame.'''
    weights /= np.sum(weights)
    '''normalizing these random numbers so that they sum to 1.0, making them proportions of the total.'''
    '''The whole process can be improve by portfolio optimization'''

    cov_matrix = returns.cov()

    tn_relation = price.shape[0] / price.shape[1]
    ked_bwidth = 0.25
    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Determine the threshold for eigenvalues
    # Here, we're using the method based on the Marcenko-Pastur distribution
    lambda_plus = (1 + np.sqrt(tn_relation)) ** 2 * ked_bwidth
    lambda_minus = (1 - np.sqrt(tn_relation)) ** 2 * ked_bwidth
    threshold = (lambda_plus + lambda_minus) / 2

    # Denoise the eigenvalues
    eigenvalues_denoised = np.where(eigenvalues < threshold, np.mean(eigenvalues[eigenvalues < threshold]), eigenvalues)

    # Reconstruct the denoised covariance matrix
    cov_matrix_denoised = eigenvectors @ np.diag(eigenvalues_denoised) @ eigenvectors.T

    port_std = np.sqrt(weights.T.dot(cov_matrix_denoised).dot(weights))
    port_ret = np.dot(returns_mean, weights)

    return returns, returns_mean, returns_std, port_std, port_ret


def port_VaR_parametric(port_ret, port_std):
    initial_investment = 1e6
    conf_level = 0.95
    alpha = norm.ppf(1 - conf_level, port_ret, port_std)
    port_VaR_param = (initial_investment - initial_investment * (1 + alpha))
    print("portfolio Parametric VaR result is {}".format(port_VaR_param))

    return port_VaR_param


if __name__ == '__main__':
    tickers = ['NWSA', 'FOXA', 'CMCSA', '^GSPC', 'LYV', 'NKE', 'NFLX', 'SONY', 'ATVI', 'CHTR', 'MAR', 'LULU', 'LVS', 'DKNG', 'DIS']
    start_date = dt.datetime(2020, 1, 1)
    end_date = dt.datetime(2023, 7, 31)
    price_ = load_raw_data(tickers, start_date, end_date)
    returns_, returns_mean_, returns_std_, port_std_, port_ret_ = de_noised_data_info(price_)
    port_VaR_param_ = port_VaR_parametric(port_ret_, port_std_)


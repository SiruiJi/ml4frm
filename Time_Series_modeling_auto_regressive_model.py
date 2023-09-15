import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import numpy as np
import datetime as dt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg


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


def lag_determine_by_pacf(train_data, tickers):
    fig, ax = plt.subplots(len(tickers), 1, figsize=(10, 6 * len(tickers)))  # create a subplot for each ticker
    plt.tight_layout()
    for idx, ticker in enumerate(tickers):
        sm.graphics.tsa.plot_pacf(train_data[ticker], lags=30, ax=ax[idx], title=ticker + ' pacf')

    plt.show()


def auto_regression_model(tickers, train_data, test_data):
    prediction = pd.DataFrame()

    for i in tickers:
        ar_model = AutoReg(train_data[i].values, lags=26)
        ar_model_fitted = ar_model.fit()
        ar_model_predict = ar_model_fitted.predict(start=len(train_data[i]),
                                                   end=len(train_data[i])+len(test_data[i])-1,
                                                   dynamic=False)

        for k in range(len(ar_model_predict)):
            print('==' * 25)
            print('predicted values:{:.4f} & actual values:{:.4f}'.format(ar_model_predict[k], test_data[i].iloc[k]))

        ar_model_predict = pd.DataFrame(ar_model_predict)
        ar_model_predict.index = test_data.index
        prediction[i] = ar_model_predict

        fig, ax = plt.subplots(figsize=(18, 15))  # create a new figure for each ticker
        ax.plot(test_data[i], label='Actual Stock Price', linestyle='--')
        ax.plot(prediction[i], label='Prediction', linestyle='solid')
        ax.set_title('Prediction Stock Price')
        ax.legend(loc='best')
        ax.set(xlabel='Date', ylabel='$')
        plt.tight_layout()
        plt.show()

    return prediction


if __name__ == '__main__':
    tickers = ['SPY', 'NVDA']
    start_date = dt.datetime(2020, 1, 1)
    end_date = dt.datetime(2023, 6, 18)
    ret_ = load_raw_data(tickers, start_date, end_date)
    stationary_test(tickers, ret_)
    train_data_, test_data_ = train_test_split(tickers, ret_)
    lag_determine_by_pacf(train_data_, tickers)
    prediction_ = auto_regression_model(tickers, train_data_, test_data_)
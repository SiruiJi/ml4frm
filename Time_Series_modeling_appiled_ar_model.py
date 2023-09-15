import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import numpy as np
import datetime as dt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg


def load_raw_data(ticker, start_date, end_date):
    price = yf.download(ticker, start_date, end_date)['Adj Close']
    ret = np.log(price / price.shift(1))
    ret.dropna(inplace=True)
    return ret


def stationary_test(ret):
        stat_test = adfuller(ret)[0:2]
        print("The ADF test statistic and p-value are {}".format(stat_test))


def train_test_split(ret):
    split = int(len(ret.values) * 0.95)

    train_data = pd.DataFrame()
    train_data = ret.iloc[:split]

    test_data = pd.DataFrame()
    test_data = ret.iloc[split:]

    return train_data, test_data


def lag_determine_by_pacf(train_data):
    fig, ax = plt.subplots(figsize=(10, 6))
    sm.graphics.tsa.plot_pacf(train_data, lags=30, ax=ax, title=' pacf')
    plt.show()


def auto_regressive_model_train(train_data, test_data, lags):
    ar = AutoReg(train_data.values, lags=lags)
    ar_fitted = ar.fit()
    ar_predict = ar_fitted.predict(start=len(train_data),
                                   end=len(train_data)+len(test_data)-1,
                                   dynamic=False)

    for i in range(len(ar_predict)):
        print('==' * 20)
        print('predicted values:{:.4f} & actual:{:.4f}'.format(ar_predict[i], test_data[i]))

    ar_predict = pd.DataFrame(ar_predict)
    ar_predict.index = test_data.index

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(test_data, label='Actual Stock Price', linestyle='--')
    ax.plot(ar_predict, label='Prediction', linestyle='solid')
    ax.set_title('Prediction Stock Price')
    ax.legend(loc='best')
    ax.set(xlabel='Date', ylabel='%')
    plt.tight_layout()
    plt.show()

    return ar_fitted, ar_predict

def next_day_prediction(ar_fitted, ret, lags):
    ret = ret.tail(lags)
    pred = ar_fitted.predict(start=len(ret), end=len(ret))
    print(f"Next day's predicted return is: {pred[0]}")
    return pred


if __name__ == '__main__':
    ticker = ['SPY']
    start_date = dt.datetime(2020, 1, 1)
    end_date = dt.datetime(2023, 6, 21)
    ret_ = load_raw_data(ticker, start_date, end_date)
    stationary_test(ret_)
    train_data_, test_data_ = train_test_split(ret_)
    lag_determine_by_pacf(train_data_)
    lags = 90
    ar_fitted_, ar_predict_ = auto_regressive_model_train(train_data_, test_data_, lags)
    pred_ = next_day_prediction(ar_fitted_, ret_, lags)

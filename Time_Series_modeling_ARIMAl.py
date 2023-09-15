import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import numpy as np
import datetime as dt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import itertools


def load_raw_data(ticker, start_date, end_date):
    price = yf.download(ticker, start_date, end_date)['Adj Close']
    ret = np.log(price / price.shift(1))
    ret = pd.DataFrame(ret)
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


def arima_order_determine(train_data):
        p = q = range(0, 9)
        d = range(0, 6)
        pdq = list(itertools.product(p, d, q))
        arima_result = []
        for param_set in pdq:
            try:
                arima = ARIMA(train_data, order=param_set)
                arima_fitted = arima.fit()
                arima_result.append(arima_fitted.aic)
            except:
                continue
        if not arima_result:  # Check if the list is empty
            print("No ARIMA model was successfully fit.")
            return None, None
        else:
            print('**' * 40)
            lowest_aic = min(arima_result)
            best_params = pdq[arima_result.index(lowest_aic)]
            print(f"The lowest AIC score is {lowest_aic:.4f} and the corresponding parameters are {best_params}")
            return lowest_aic, best_params


def arima_model(train_data, test_data):
    arima = ARIMA(train_data, order=(20, 0, 20))
    arima_fit = arima.fit()
    arima_predict = arima_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)
    arima_predict = pd.DataFrame(arima_predict)
    arima_predict.index = test_data.index

    fig, ax = plt.subplots(figsize=(18, 15))
    ax.plot(test_data, label='Actual Stock Price', linestyle='--')
    ax.plot(arima_predict, label='Prediction', linestyle='solid')
    ax.set_title('Predicted Stock Price')
    ax.legend(loc='best')
    plt.show()
    print(arima_fit.summary())

    return arima_fit, arima_predict


if __name__ == '__main__':
    ticker = ['SPY']
    start_date = '2015-01-01'
    end_date = '2016-01-01'
    ret_ = load_raw_data(ticker, start_date, end_date)
    stationary_test(ret_)
    train_data_, test_data_ = train_test_split(ret_)
    lowest_aic, best = arima_order_determine(train_data_)
    #arima_fit_, arima_predict = arima_model(train_data_, test_data_)

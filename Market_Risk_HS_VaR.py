import numpy as np
import yfinance as yf
import datetime as dt
from arch import arch_model
import matplotlib.pyplot as plt


def load_raw_data(ticker, start_date, end_date):
    price = yf.download(ticker, start_date, end_date)['Adj Close']
    ret =100 * price.pct_change()[1:]
    ret.dropna(inplace=True)
    return ret


def var_calculation(ret):
    var1_hs = np.empty(len(ret))

    for i in range(500, len(ret)):
        retwindow = ret[(i - 500):(i - 1)]

        var1_hs[i] = -np.percentile(retwindow, 1)  # 1 for i percent

    plt.plot(var1_hs, color='blue')
    plt.show()

    return var1_hs


if __name__ == '__main__':
    ticker = '^GSPC'
    start_date = dt.datetime(2016, 1, 1)
    end_date = dt.datetime(2023, 8, 1)
    ret_ = load_raw_data(ticker, start_date, end_date)
    var1_hs_ = var_calculation(ret_)

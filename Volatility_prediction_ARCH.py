import numpy as np
import yfinance as yf
import datetime as dt
from arch import arch_model
import matplotlib.pyplot as plt


def load_raw_data(ticker, start_date, end_date):
    price = yf.download(ticker, start_date, end_date)['Adj Close']
    ret = 100 * price.pct_change()[1:]
    return ret


def model_train(ret):
    global best_param
    bic_arch = []
    for p in range(1, 5):
        arch = arch_model(ret, mean='zero', vol='ARCH', p=p).fit(disp='off')
        bic_arch.append(arch.bic)
        if arch.bic == np.min(bic_arch):
            best_param = p
    arch = arch_model(ret, mean='zero', vol='ARCH', p=best_param).fit(disp='off')
    print(arch.summary())

    realized_vol = ret.rolling(5).std()
    n = 252
    split_date = ret.iloc[-n:].index
    forecast = arch.forecast(start=split_date[0])
    plt.figure(figsize=(10, 6))
    plt.plot(realized_vol / 100, label='Realized Volatility')
    plt.plot(forecast.variance.iloc[-len(split_date):] / 100, label='Volatility Prediction-ARCH')
    plt.title('Volatility Prediction with ARCH')
    plt.legend()
    plt.show()
    return arch


if __name__ == '__main__':
    ticker = '^GSPC'
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2023, 7, 11)
    ret_ = load_raw_data(ticker, start_date, end_date)
    arch_model_ = model_train(ret_)
    forecasts = arch_model_.forecast(start=0)
    next_day_volatility = np.sqrt(forecasts.variance.iloc[-1, :] / 100)
    print(next_day_volatility)

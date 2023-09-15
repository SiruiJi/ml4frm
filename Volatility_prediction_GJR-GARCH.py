import numpy as np
import yfinance as yf
import datetime as dt
from arch import arch_model
import matplotlib.pyplot as plt


def load_raw_data(ticker, start_date, end_date):
    price = yf.download(ticker, start_date, end_date)['Adj Close']
    ret = 100 * price.pct_change()[1:]
    ret.dropna(inplace=True)
    return ret


def model_train(ret):

    global gjrgarch, q, best_param
    bic_gjr_garch = []
    for p in range(1, 5):
        for q in range(1, 5):
            gjrgarch = arch_model(ret, mean='zero', p=p, o=1, q=q).fit(disp='off')
        bic_gjr_garch.append(gjrgarch.bic)
        if gjrgarch.bic == np.min(bic_gjr_garch):
            best_param = p, q
    gjrgarch = arch_model(ret, mean='zero', p=best_param[0], o=1, q=best_param[1]).fit(disp='off')
    print(gjrgarch.summary())

    realized_vol = ret.rolling(5).std()
    n = 252
    split_date = ret.iloc[-n:].index
    forecast = gjrgarch.forecast(start=split_date[0])
    plt.figure(figsize=(10, 6))
    plt.plot(realized_vol / 100, label='Realized Volatility')
    plt.plot(forecast.variance.iloc[-len(split_date):] / 100, label='Volatility Prediction-GJR_GARCH')
    plt.title('Volatility Prediction with GJR-GARCH')
    plt.legend()
    plt.show()
    return gjrgarch


if __name__ == '__main__':
    ticker = '^GSPC'
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2021, 8, 1)
    ret_ = load_raw_data(ticker, start_date, end_date)
    gjrgarch_model_ = model_train(ret_)
    forecasts = gjrgarch_model_.forecast(start=0)
    next_day_volatility = np.sqrt(forecasts.variance.iloc[-1, :] / 100)
    print(next_day_volatility)
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

    global egarch, q, best_param
    bic_egarch = []
    for p in range(1, 5):
        for q in range(1, 5):
            egarch = arch_model(ret, mean='zero', vol='EGARCH', p=p, q=q).fit(disp='off')
        bic_egarch.append(egarch.bic)
        if egarch.bic == np.min(bic_egarch):
            best_param = p, q
    egarch = arch_model(ret, mean='zero', vol='EGARCH', p=best_param[0], q=best_param[1]).fit(disp='off')
    print(egarch.summary())

    realized_vol = ret.rolling(5).std()
    n = 252
    split_date = ret.iloc[-n:].index
    forecast = egarch.forecast(start=split_date[0])
    plt.figure(figsize=(10, 6))
    plt.plot(realized_vol / 100, label='Realized Volatility')
    plt.plot(forecast.variance.iloc[-len(split_date):] / 100, label='Volatility Prediction-EGARCH')
    plt.title('Volatility Prediction with EGARCH')
    plt.legend()
    plt.show()
    return egarch


if __name__ == '__main__':
    ticker = '^GSPC'
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2021, 8, 1)
    ret_ = load_raw_data(ticker, start_date, end_date)
    egarch_model_ = model_train(ret_)
    forecasts = egarch_model_.forecast(start=0)
    next_day_volatility = np.sqrt(forecasts.variance.iloc[-1, :] / 100)
    print(next_day_volatility)
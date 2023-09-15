import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from sklearn.metrics import mean_squared_error as mse


def load_raw_data(ticker, start_date, end_date):
    price = yf.download(ticker, start_date, end_date, interval='1d')['Adj Close']
    ret = 100 * price.pct_change()[1:]
    return ret


def data_preparation(ret):
    realized_vol = ret.rolling(5).std()
    realized_vol = pd.DataFrame(realized_vol)
    original_index = realized_vol.index[4:]
    realized_vol.reset_index(drop=True, inplace=True)

    returns_dl = ret ** 2
    returns_dl = returns_dl.reset_index()
    del returns_dl['Date']

    X = pd.concat([realized_vol, returns_dl], axis=1, ignore_index=True)
    X = X[4:].copy()
    X = X.reset_index()
    X.drop('index', axis=1, inplace=True)

    realized_vol = realized_vol.dropna().reset_index()
    realized_vol.drop('index', axis=1, inplace=True)
    return realized_vol, returns_dl, X, original_index



def deep_learning_model_train(X, realized_vol, ret, original_index):
    model = keras.Sequential(
        [layers.Dense(256, activation="relu"),
         layers.Dense(128, activation="relu"),
         layers.Dense(1, activation="linear"), ])

    model.compile(loss='mse', optimizer='rmsprop')

    n = 252
    epochs_trial = np.arange(100, 400, 4)
    batch_trial = np.arange(100, 400, 4)
    DL_pred = []
    DL_RMSE = []
    for i, j, k in zip(range(4), epochs_trial, batch_trial):
        model.fit(X.iloc[:-n],
                  realized_vol.iloc[1:-(n-1)].values.reshape(-1,),
                  batch_size=k, epochs=j, verbose=False)
        DL_predict = model.predict(np.asarray(X.iloc[-n:]))
        DL_RMSE.append(np.sqrt(mse(realized_vol.iloc[-n:] / 100,
                                   DL_predict.flatten() / 100)))
        DL_pred.append(DL_predict)
        print('DL_RMSE_{}:{:.6f}'.format(i+1, DL_RMSE[i]))

    DL_predict = pd.DataFrame(DL_pred[DL_RMSE.index(min(DL_RMSE))])
    DL_predict.index = ret.iloc[-n:].index
    realized_vol.index = original_index


    plt.figure(figsize=(10, 6))
    plt.plot(realized_vol / 100, label='Realized Volatility')
    plt.plot(DL_predict / 100, label='Volatility Prediction-DL')
    plt.title('Volatility Prediction with Deep Learning', fontsize=12)
    plt.legend()
    plt.show()
    return model, realized_vol, DL_predict


def next_day_prediction(model):
    price = yf.download('^GSPC', start='2023-06-29', end='2023-07-10', interval='1d')['Adj Close']
    pret = 100 * price.pct_change()
    vol = pret.rolling(5).std()
    vol = pd.DataFrame(vol)
    vol.reset_index(drop=True, inplace=True)
    ret_dl = pret ** 2
    ret_dl = ret_dl.reset_index()
    del ret_dl['Date']
    pX = pd.concat([vol, ret_dl], axis=1, ignore_index=True)
    pX = pX[4:].copy()
    pX = pX.reset_index()
    pX.drop('index', axis=1, inplace=True)
    pX.dropna(inplace=True)

    vol = vol.dropna().reset_index()
    vol.drop('index', axis=1, inplace=True)

    pred = model.predict(pX)

    return vol, ret_dl, pX, pred


if __name__ == '__main__':
    ticker = '^GSPC'
    start_date = dt.datetime(2018, 1, 1)
    end_date = dt.datetime(2023, 1, 1)
    ret_ = load_raw_data(ticker, start_date, end_date)
    realized_vol_, returns_dl, X_, original_index_ = data_preparation(ret_)
    model_, realized_vol_, DL_predict_ = deep_learning_model_train(X_, realized_vol_, ret_, original_index_)
    vol_, ret_dl_, pX_, pred_ = next_day_prediction(model_)

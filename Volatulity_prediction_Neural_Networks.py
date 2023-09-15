import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error as mse


def load_raw_data(ticker, start_date, end_date):
    price = yf.download(ticker, start_date, end_date, interval='1d')['Adj Close']
    ret = 100 * price.pct_change()[1:]
    return ret


def data_preparation(ret):
    realized_vol = ret.rolling(5).std()
    realized_vol = pd.DataFrame(realized_vol)
    realized_vol.reset_index(drop=True, inplace=True)

    returns_svm = ret ** 2
    returns_svm = returns_svm.reset_index()
    del returns_svm['Date']

    X = pd.concat([realized_vol, returns_svm], axis=1, ignore_index=True)
    X = X[4:].copy()
    X = X.reset_index()
    X.drop('index', axis=1, inplace=True)

    realized_vol = realized_vol.dropna().reset_index()
    realized_vol.drop('index', axis=1, inplace=True)
    return realized_vol, returns_svm, X


def neural_networks_train(X, realized_vol, ret):
    NN_vol = MLPRegressor(learning_rate_init=0.001, random_state=1)
    para_grid_NN = {'hidden_layer_sizes': [(100, 50), (50, 50), (10, 100)],
                    'max_iter': [500, 1000],
                    'alpha': [0.00005, 0.0005]}
    n = 252
    model = RandomizedSearchCV(NN_vol,para_grid_NN)
    model.fit(X.iloc[:-n].values,
            realized_vol.iloc[1:-(n-1)].values.reshape(-1,))
    NN_predictionss = model.predict(X.iloc[-n:])

    NN_predictionss = pd.DataFrame(NN_predictionss)
    NN_predictionss.index = ret.iloc[-n:].index

    rmse_NN = np.sqrt(mse(realized_vol.iloc[-n:] / 100, NN_predictionss / 100))
    print('The RMSE value of SVR with Linear Kernel is {:.6f}'.format(rmse_NN))

    realized_vol.index = ret.iloc[4:].index

    plt.figure(figsize=(10, 6))
    plt.plot(realized_vol / 100, label='Realized Volatility')
    plt.plot(NN_predictionss / 100, label='Volatility Prediction-NN')
    plt.title('Volatility Prediction with Neural Network', fontsize=12)
    plt.legend()
    plt.show()

    return model, NN_predictionss


def next_day_prediction(ret, model):
    price = yf.download('^GSPC', start='2023-06-30', end='2023-07-12', interval='1d')['Adj Close']
    pret = 100 * price.pct_change()
    vol = pret.rolling(5).std()
    vol = pd.DataFrame(vol)
    vol.reset_index(drop=True, inplace=True)
    ret_svm = pret ** 2
    ret_svm = ret_svm.reset_index()
    del ret_svm['Date']
    pX = pd.concat([vol, ret_svm], axis=1, ignore_index=True)
    pX = pX[4:].copy()
    pX = pX.reset_index()
    pX.drop('index', axis=1, inplace=True)
    pX.dropna(inplace=True)

    vol = vol.dropna().reset_index()
    vol.drop('index', axis=1, inplace=True)

    pred = model.predict(pX)

    return vol, ret_svm, pX, pred


if __name__ == '__main__':
    ticker = '^GSPC'
    start_date = dt.datetime(2018, 1, 1)
    end_date = dt.datetime(2023, 6, 1)
    ret_ = load_raw_data(ticker, start_date, end_date)
    realized_vol_, returns_svm_, X_ = data_preparation(ret_)
    model_, predict = neural_networks_train(X_, realized_vol_, ret_)
    vol_, ret_svm_, pX, pred_ = next_day_prediction(ret_, model_)
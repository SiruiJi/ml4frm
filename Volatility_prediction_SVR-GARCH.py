import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from scipy.stats import uniform as sp_rand
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


def svm_model_train(X, realized_vol, ret):
    svr_lin = SVR(kernel='linear')
    '''The alternative kernel are (rbf) and (poly, degree=2)'''
    n = 252
    para_grid = {'gamma': sp_rand(),
                 'C': sp_rand(),
                 'epsilon': sp_rand()}

    model = RandomizedSearchCV(svr_lin, para_grid)
    model.fit(X.iloc[:-n].values,
            realized_vol.iloc[1:-(n-1)].values.reshape(-1,))
    predict_svr_lin = model.predict(X.iloc[-n:])

    predict_svr_lin = pd.DataFrame(predict_svr_lin)
    predict_svr_lin.index = ret.iloc[-n:].index

    rmse_svr = np.sqrt(mse(realized_vol.iloc[-n:] / 100, predict_svr_lin / 100))
    print('The RMSE value of SVR with Linear Kernel is {:.6f}'.format(rmse_svr))

    realized_vol.index = ret.iloc[4:].index

    plt.figure(figsize=(10, 6))
    plt.plot(realized_vol / 100, label='Realized Volatility')
    plt.plot(predict_svr_lin / 100, label='Volatility Prediction-SRV-GARCH')
    plt.title('Volatility Prediction with SVR-GARCH (Linear)', fontsize=12)
    plt.legend()
    plt.show()

    return model, predict_svr_lin

def next_day_prediction(ret, model):
    price = yf.download('^GSPC', start='2023-05-08', end='2023-05-16', interval='1d')['Adj Close']
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
    end_date = dt.datetime(2023, 1, 1)
    ret_ = load_raw_data(ticker, start_date, end_date)
    realized_vol_, returns_svm_, X_ = data_preparation(ret_)
    model_, predict = svm_model_train(X_, realized_vol_, ret_)
    vol_, ret_svm_, pX, pred_ = next_day_prediction(ret_, model_)

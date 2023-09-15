import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import (Dense, Dropout, Flatten, LSTM)


def load_raw_data(ticker, start_date, end_date):
    price = yf.download(ticker, start_date, end_date)['Adj Close']
    ret = price.diff().dropna()
    ret = pd.DataFrame(ret)
    return ret


def train_test_split(ret):
    split = int(len(ret.values) * 0.95)
    train_data = ret.iloc[:split]
    test_data = ret.iloc[split:]
    return train_data, test_data


def split_sequence(sequence, n_steps):
    sequence = sequence.values.flatten()
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def train_rnn_model(n_steps, n_features, train_data, test_data):
    model = Sequential()
    model.add(LSTM(512, activation='relu', input_shape=(n_steps, n_features), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mse'])

    X, y = split_sequence(train_data, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    history = model.fit(X, y, epochs=400, batch_size=150, verbose=0, validation_split=0.1)

    start = X[X.shape[0] - n_steps]
    x_input = start
    x_input = x_input.reshape((1, n_steps, n_features))

    tempList = []
    for i in range(len(test_data)):
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        x_input = np.append(x_input, yhat)
        x_input = x_input[1:]
        tempList.append(yhat)

    fig, ax = plt.subplots(figsize=(18, 15))
    ax.plot(test_data, label='Actual Stock Price', linestyle='--')

    ax.plot(test_data.index, np.array(tempList).flatten(), label='Prediction', linestyle='solid')
    ax.set_title('Prediction Stock Price')
    ax.legend(loc='best')
    plt.show()
    return model


def predict_next_day_return(model, n_steps, ret):
    # Get the data for the last n_steps days
    last_n_days_data = ret.tail(n_steps)
    # Get the last n_steps data points
    x_input = last_n_days_data.values.flatten()[-n_steps:]
    # Reshape it for the model
    x_input = x_input.reshape((1, n_steps, 1))
    # Predict the next day return
    predicted_return = model.predict(x_input, verbose=0)
    return predicted_return[0][0]


if __name__ == '__main__':
    ticker = 'AAPL'
    start_date = dt.datetime(2019, 1, 1)
    end_date = dt.datetime(2020, 1, 1)
    ret_ = load_raw_data(ticker, start_date, end_date)
    train_data_, test_data_ = train_test_split(ret_)

    n_steps = 13
    n_features = 1
    model_ = train_rnn_model(n_steps, n_features, train_data_, test_data_)
    predicted_return_ = predict_next_day_return(model_, n_steps, ret_)
    print(predicted_return_)


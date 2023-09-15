import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import datetime as dt
from sklearn.linear_model import LinearRegression


def load_raw_data(tickers, start_date, end_date):
    raw = yf.download(tickers, start_date, end_date)['Adj Close']
    raw = raw.reindex(columns=tickers)
    raw = pd.DataFrame(raw)
    return raw


def rolling_data(raw, symbol, window):
    data = pd.DataFrame(raw[symbol], columns=[symbol])
    data['ma'] = raw[symbol].rolling(window).mean()
    data['returns'] = data['ma'].pct_change()
    data['mareturns'] = data['returns'].rolling(window).mean() * 252
    data['volatility'] = data['returns'].rolling(window).std() * 252
    data.dropna(inplace=True)
    return data


def linear_relation(data):
    model = LinearRegression()
    X = data[['volatility']].dropna() # Reshape data and drop NA values
    Y = data['mareturns'].dropna()
    model.fit(X, Y)  # Fit the model
    data['prediction'] = model.predict(X)
    fig = go.Figure()
    fig.add_trace(go.Scatter(name='Risk-Return Relationship',x=data['volatility'], y=data['mareturns'], mode='markers'))
    fig.add_trace(go.Scatter(name='Best Fit Line', x=data['volatility'], y=data['prediction'], mode='lines'))
    fig.update_layout(xaxis_title='Standard Deviation', yaxis_title='Return', width=900, height=470)
    fig.show()
    return data


if __name__ == '__main__':
    tickers = ['SPY', 'NVDA']
    start_date = dt.datetime(2020, 1, 1)
    end_date = dt.datetime(2023, 6, 16)
    raw_ = load_raw_data(tickers, start_date, end_date)
    symbol = 'NVDA'
    window = 30
    data_ = rolling_data(raw_, symbol, window)
    data_ = linear_relation(data_)
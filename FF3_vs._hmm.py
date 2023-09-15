import pandas as pd
import yfinance as yf
import datetime as dt
from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
from matplotlib import cm
import statsmodels.api as sm


def load_raw_data(file_path, ticker, start_date, end_date):
    ff = pd.read_csv(file_path)
    ff = ff.rename(columns={'Unnamed: 0': 'Date'})
    print(ff.head())
    print(ff.info())
    ff['Date'] = pd.to_datetime(ff['Date'], format='%Y%m%d')
    ff.set_index('Date', inplace=True)
    ff_trim = ff.loc['2000-01-01':]
    print(ff_trim.head())
    print(ff_trim.info())

    sp_etf = yf.download(ticker, start_date, end_date)
    sp = pd.DataFrame()
    sp['Close'] = sp_etf['Adj Close']
    sp['return'] = (sp['Close'] / sp['Close'].shift(1)) - 1

    ff_merge = pd.merge(ff_trim, sp['return'], how='inner', on='Date')
    return ff_merge, sp


def optimum_number_of_states(sp):
    sp_ret = sp['return'].dropna().values.reshape(-1, 1)
    n_components = np.arange(1, 10)
    clusters = [hmm.GaussianHMM(n_components=n, covariance_type="full", random_state=123).fit(sp_ret) for n in n_components]
    plt.plot(n_components, [m.score(np.array(sp['return'].dropna()).reshape(-1, 1)) for m in clusters])
    plt.title('Optimum Number of States')
    plt.xlabel('n_components')
    plt.ylabel('Log Likelihood')
    plt.show()


def predict_hidden_state(sp):
    hmm_model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=123)
    hmm_model.fit(np.array(sp['return'].dropna()).reshape(-1, 1))
    hmm_predict = hmm_model.predict(np.array(sp['return'].dropna()).reshape(-1, 1))
    df_hmm = pd.DataFrame(hmm_predict)

    ret_merged = pd.concat([df_hmm, sp['return'].dropna().reset_index()], axis=1)
    ret_merged.drop('Date', axis=1, inplace=True)
    ret_merged.rename(columns={0: 'states'}, inplace=True)
    print(ret_merged.dropna().head())
    print(ret_merged['states'].value_counts())

    state_means = []
    state_std = []

    for i in range(3):
        state_means.append(ret_merged[ret_merged['states'] == i]['return'].mean())
        state_std.append(ret_merged[ret_merged['states'] == i]['return'].std())
    print('State Means are:', ', '.join('{:.4f}'.format(mean) for mean in state_means))
    print('State Standard Deviations are:', ', '.join('{:.4f}'.format(std) for std in state_std))

    print(f'HMM means\n {hmm_model.means_}')
    print(f'HMM covariances \n {hmm_model.covars_}')
    print(f'HMM transition matrix\n {hmm_model.transmat_}')
    print(f'HMM initial probability\n {hmm_model.startprob_}')

    df_sp_ret = sp['return'].dropna()
    sp_ret = sp['return'].dropna().values.reshape(-1, 1)
    hidden_states = hmm_model.predict(sp_ret)
    fig, axs = plt.subplots(hmm_model.n_components, sharex=True, sharey=True, figsize=(12, 9))
    colors = cm.gray(np.linspace(0, 0.7, hmm_model.n_components))
    for i, (ax, color) in enumerate(zip(axs, colors)):
        mask = hidden_states == i
        ax.plot_date(df_sp_ret.index.values[mask],
                     df_sp_ret.values[mask],
                     ".-", c=color)
        ax.set_title("Hidden state {}".format(i + 1), fontsize=16)
        ax.xaxis.set_minor_locator(MonthLocator())
    plt.tight_layout()
    plt.show()

    print(ret_merged.groupby('states')['return'].mean())
    return df_hmm



def train_test_split(ff_merge):
    split = int(len(ff_merge) * 0.9)
    train_ff = ff_merge[:split].dropna()
    test_ff = ff_merge[split:].dropna()
    return train_ff, test_ff


def gaussian_hmm_model(train_ff, test_ff):
    hmm_model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100, init_params="")

    prediction = []
    for i in range(len(test_ff)):
        hmm_model.fit(train_ff)
        adjustment = np.dot(hmm_model.transmat_, hmm_model.means_)
        prediction.append(test_ff.iloc[i] + adjustment[0])
    prediction = pd.DataFrame(prediction)

    std_dev = prediction['return'].std()
    sharpe = prediction['return'].mean() / std_dev
    print('Sharpe ratio with HMM is {:.4f}'.format(sharpe))
    plt.plot(prediction.index, prediction['return'], color='blue')
    plt.plot(prediction.index,  test_ff['return'], color='red')
    plt.show()

    return prediction


def ff3_model(train_ff, test_ff):
    Y = train_ff['return']
    X = train_ff[['Mkt-RF', 'SMB', 'HML']]

    model = sm.OLS(Y, X)
    ff_ols = model.fit()
    print(ff_ols.summary())

    ff_pred = ff_ols.predict(test_ff[['Mkt-RF', 'SMB', 'HML']])
    print(ff_pred.head())

    std_dev = ff_pred.std()
    sharpe = ff_pred.mean() / std_dev
    print('Sharpe ratio with FF 3 factor model is {:.4f}'.format(sharpe))


if __name__ == '__main__':
    filepath = 'D:/PyCharm Community Edition 2023.1.2/Python_Project/Finance/ml4frm/F-F_Research_Data_Factors_daily.csv'
    ticker = 'SPY'
    start_date = dt.datetime(2000, 1, 3)
    end_date = dt.datetime(2023, 8, 1)
    ff_merge_, sp_ = load_raw_data(filepath, ticker, start_date, end_date)
    #optimum_number_of_states(sp_)
    #df_hmm_ = predict_hidden_state(sp_)
    train_ff_, test_ff_ = train_test_split(ff_merge_)
    prediction_ = gaussian_hmm_model(train_ff_, test_ff_)
    #ff3_model(train_ff_, test_ff_)



import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from copulae.mixtures.gmc.gmc import GaussianMixtureCopula

def load_raw_data(file_path):
    liq_data = pd.read_csv(file_path)
    liq_data = liq_data[~(liq_data == '#VALUE!').any(axis=1)]
    for col in ['Open', 'ASKHI', 'BIDLO', 'Close', 'PRC', 'Volume', 'RET', 'SHROUT', 'TP', 'vwretx']:
        liq_data[col] = pd.to_numeric(liq_data[col], errors='coerce')

    rolling_window = liq_data.rolling(window=5)
    liq_data['bidlo_min'] = rolling_window['BIDLO'].min()
    liq_data['askhi_max'] = rolling_window['ASKHI'].max()
    liq_data['vol_sum'] = rolling_window['Volume'].sum()
    liq_data['shrout_mean'] = rolling_window['SHROUT'].mean()
    liq_data['prc_mean'] = rolling_window['PRC'].mean()
    return liq_data


def percentage_quoted_and_effective_bid_ask_spread(liq_vol_all):

    mid_price = (liq_vol_all.ASKHI + liq_vol_all.BIDLO) / 2

    percent_quote_ba = (liq_vol_all.ASKHI - liq_vol_all.BIDLO) / mid_price
    percent_effective_ba = 2 * abs((liq_vol_all.PRC - mid_price)) / mid_price

    return percent_quote_ba, percent_effective_ba


def rolls_spread_estimate(liq_vol_all):
    price_diff = liq_vol_all.PRC.diff()
    price_diff.dropna(inplace=True)

    roll = []
    for i in range(len(price_diff)-5):
        roll_cov = np.cov(price_diff[i:i+5], price_diff[i+1:i+6])
        if roll_cov[0, 1] < 0:
            roll.append(2 * np.sqrt(-roll_cov[0, 1]))
        else:
            roll.append(2 * np.sqrt(np.abs(roll_cov[0, 1])))
    roll = pd.DataFrame(roll)

    return roll


def corwin_schultz_spread(liq_vol_all):
    global beta_array, gamma_array
    gamma = []
    for i in range(len(liq_vol_all)-1):
        gamma.append((max(liq_vol_all['ASKHI'].iloc[i+1],
                          liq_vol_all['ASKHI'].iloc[i]) -
                      min(liq_vol_all['BIDLO'].iloc[i+1],
                          liq_vol_all['BIDLO'].iloc[i])) ** 2)
        gamma_array = np.array(gamma)

    beta = []
    for i in range(len(liq_vol_all)-1):
        beta.append((liq_vol_all['ASKHI'].iloc[i+1] - liq_vol_all['BIDLO'].iloc[i+1]) ** 2 +
                    (liq_vol_all['ASKHI'].iloc[i] - liq_vol_all['BIDLO'].iloc[i]) ** 2)
        beta_array = np.array(beta)

    alpha = ((np.sqrt(2 * beta_array) - np.sqrt(beta_array)) / (3 - (2 * np.sqrt(2)))) - np.sqrt(gamma_array / (3 - (2 * np.sqrt(2))))
    cs_spread = (2 * np.exp(alpha - 1)) / (1 + np.exp(alpha))
    cs_spread = pd.DataFrame(cs_spread)
    return cs_spread

def liquidity_ratio(liq_vol_all):
    liq_ratio = []
    for i in range(len(liq_vol_all)):
        liq_ratio.append(liq_vol_all['PRC'][i+1:i+6].sum() * liq_vol_all['Volume'][i+1:i+6].sum() /
                         (np.abs(liq_vol_all['PRC'][i+1:i+6].mean() - liq_vol_all['PRC'][i:i+5].mean())))
    liq_ratio = pd.DataFrame(liq_ratio)

    return liq_ratio


def hui_heubel_ratio(liq_vol_all):
    lhh = []
    for i in range(len(liq_vol_all)):
        lhh.append((liq_vol_all['PRC'][i:i+5].max() - liq_vol_all['PRC'][i:i+5].min()) /
                   liq_vol_all['PRC'][i:i+5].min() /
                   (liq_vol_all['Volume'][i:i+5].sum() /
                    liq_vol_all['SHROUT'][i:i+5].mean() *
                    liq_vol_all['PRC'][i:i+5].mean()))
    lhh = pd.DataFrame(lhh)
    return lhh


def turn_over_ratio(liq_vol_all):
    turnover_ratio = []
    for i in range(len(liq_vol_all)):
        turnover_ratio.append((1/liq_vol_all['Volume'].count()) *
                              (liq_vol_all['Volume'][i:i+1].sum() /
                               liq_vol_all['SHROUT'][i:i+1].sum()))
    turnover_ratio = pd.DataFrame(turnover_ratio)
    return turnover_ratio


def amihud_illiquidity(liq_vol_all):
    dvol = []
    for i in range(len(liq_vol_all)):
        dvol.append((liq_vol_all['PRC'][i:i+5] * liq_vol_all['Volume'][i:i+5]).sum())
    dvol = pd.DataFrame(dvol)

    amihud = []
    for i in range(len(liq_vol_all)):
        amihud.append((1 / liq_vol_all['RET'].count()) *
                      (np.sum(np.abs(liq_vol_all['RET'][i:i+1])) /
                       np.sum(dvol[i:i+1])))
    amihud = pd.DataFrame(amihud)

    return amihud


def return_to_turnover(liq_vol_all):
    turnover_ratio = []
    for i in range(len(liq_vol_all)):
        turnover_ratio.append((1 / liq_vol_all['Volume'].count()) *
                              (liq_vol_all['Volume'][i:i + 1].sum() /
                               liq_vol_all['SHROUT'][i:i + 1].sum()))
    florackis = []
    for i in range(len(liq_vol_all)):
        florackis.append((1 / liq_vol_all['RET'].count()) *
                         (np.sum(np.abs(liq_vol_all['RET'][i:i+1]) /
                                 turnover_ratio[i:i+1])))
    florackis = pd.DataFrame(florackis)

    return florackis


def coefficient_of_elasticity(liq_vol_all):
    vol_diff_pct = liq_vol_all['Volume'].diff().pct_change().dropna()
    price_diff_pct = liq_vol_all['PRC'].diff().pct_change().dropna()
    cet = []
    for i in range(len(vol_diff_pct)):
        vol_pct = vol_diff_pct.iloc[i]
        price_pct = price_diff_pct.iloc[i]
        if np.isnan(vol_pct) or np.isnan(price_pct):
            cet.append(np.nan)
            continue
        constant = 1e-9
        elasticity = vol_pct / (price_pct + constant)
        cet.append(elasticity)
    cet_df = pd.DataFrame(cet, columns=['Elasticity'])
    return cet_df

def market_based_measures(liq_vol_all):
    liq_vol_all = liq_vol_all.dropna(subset=['vwretx', 'RET', 'Volume'])
    vol_pct_change = liq_vol_all['Volume'].pct_change().dropna()

    X1 = sm.add_constant(liq_vol_all['vwretx'])
    y1 = liq_vol_all['RET']
    ols1 = sm.OLS(y1, X1).fit()
    unsys_resid = ols1.resid

    unsys_resid = unsys_resid.iloc[1:]

    X2 = sm.add_constant(vol_pct_change)
    y2 = unsys_resid ** 2
    ols2 = sm.OLS(y2, X2).fit()
    market_impact = ols2.resid

    return ols2.summary(), market_impact


def all_liquidity_measures(percent_quote_ba, percent_effective_ba, roll, cs_spread, liq_ratio, lhh, turnover_ratio, amihud, florackis, cet, market_impact):
    liq_measures_dict = {
        'Percent_Quoted_BA': percent_quote_ba,
        'Percent_Effective_BA': percent_effective_ba,
        'Roll': roll,
        'CS_Spread': cs_spread,
        'Liq_Ratio': liq_ratio,
        'LHH': lhh,
        'Turnover_Ratio': turnover_ratio,
        'Amihud': amihud,
        'Florackis': florackis,
        'CET': cet,
        'Market_Impact': market_impact
    }
    liq_measures_all = pd.concat(liq_measures_dict, axis=1)
    liq_measures_all.dropna(how='all', inplace=True)
    summary_stats = liq_measures_all.describe().T
    return liq_measures_all, summary_stats


def cluster_test(liq_measures_all):
    liq_measures_all2 = liq_measures_all.dropna()
    scaled_liq = StandardScaler().fit_transform(liq_measures_all2)

    kwargs = dict(alpha=0.5, bins=50, stacked=True)
    plt.hist(liq_measures_all.loc[:, 'Percent_Quoted_BA'],
             **kwargs, label='TC-based')
    plt.hist(liq_measures_all.loc[:, 'Turnover_Ratio'],
             **kwargs, label='Volume-based')
    plt.hist(liq_measures_all.loc[:, 'Market_Impact'],
             **kwargs, label='Market-based')
    plt.title('Multi Modality of the Liquidity Measures')
    plt.legend()
    plt.show()

    n_components = np.arange(1, 10)
    clusters = [GaussianMixture(n, covariance_type='spherical', random_state=0).fit(scaled_liq)
                for n in n_components]
    plt.plot(n_components, [m.bic(scaled_liq) for m in clusters])
    plt.title('Optimum Number of Components')
    plt.xlabel('n_components')
    plt.ylabel('BIC values')
    plt.show()

    return scaled_liq


def cluster_state(scaled_liq, nstates):
    gmm = GaussianMixture(n_components=nstates,
                          covariance_type='spherical',
                          init_params='kmeans')

    gmm_fit = gmm.fit(scaled_liq)
    labels = gmm_fit.predict(scaled_liq)
    state_probs = gmm.predict_proba(scaled_liq)
    state_probs_df = pd.DataFrame(state_probs, columns=['state-1', 'state-2', 'state-3'])
    state_probs_means = [state_probs_df.iloc[:, i].mean() for i in range(len(state_probs_df.columns))]

    if np.max(state_probs_means) == state_probs_means[0]:
        print('state-1 is likely to occur with a probability of {:4f}'.format(state_probs_means[0]))
    elif np.max(state_probs_means) == state_probs_means[1]:
        print('state-2 is likely to occur with a probability of {:4f}'.format(state_probs_means[1]))
    else:
        print('state-3 is likely to occur with a probability of {:4f}'.format(state_probs_means[2]))
    return state_probs


def pca_test(scaled_liq):
    pca = PCA(n_components=11)
    components = pca.fit_transform(scaled_liq)
    plt.plot(pca.explained_variance_ratio_)
    plt.title('Screen Plot')
    plt.xlabel('Number of Components')
    plt.ylabel('% of Explained Variance')
    plt.show()

def gmm_pca(scaled_liq):
    pca= PCA(n_components=2)
    components = pca.fit_transform(scaled_liq)
    mxtd = GaussianMixture(n_components=2, covariance_type='spherical')
    gmm = mxtd.fit(components)
    labels = gmm.predict(components)
    state_probs = gmm.predict_proba(components)
    return state_probs, pca


def gmcm_test(scaled_liq):
    _, dim = scaled_liq.shape
    gmcm = GaussianMixtureCopula(n_clusters=3, ndim=dim)
    gmcm_fit = gmcm.fit(scaled_liq,method='kmeans', criteria='GMCM', eps=0.0001)
    state_probs = gmcm_fit.params.prob
    print(f'The state {np.argmax(state_probs) + 1} is likely to occur')
    print(f'State probabilities based on GMCM are {state_probs}')


if __name__ == '__main__':
    file_path = 'D:/PyCharm Community Edition 2023.1.2/Python_Project/Finance/ml4frm/AMD.csv'
    liq_vol_all_ = load_raw_data(file_path)
    percent_quote_ba_, percent_effective_ba_ = percentage_quoted_and_effective_bid_ask_spread(liq_vol_all_)
    roll_ = rolls_spread_estimate(liq_vol_all_)
    cs_spread_ = corwin_schultz_spread(liq_vol_all_)
    liq_ratio_ = liquidity_ratio(liq_vol_all_)
    lhh_ = hui_heubel_ratio(liq_vol_all_)
    turnover_ratio_ = turn_over_ratio(liq_vol_all_)
    amihud_ = amihud_illiquidity(liq_vol_all_)
    florackis_ = return_to_turnover(liq_vol_all_)
    cet_ = coefficient_of_elasticity(liq_vol_all_)
    market_impact_summary, market_impact_ = market_based_measures(liq_vol_all_)
    liq_measures_all_, summary_ = all_liquidity_measures(percent_quote_ba_, percent_effective_ba_, roll_, cs_spread_, liq_ratio_, lhh_, turnover_ratio_, amihud_, florackis_, cet_, market_impact_)
    print(summary_)
    scaled_liq_ = cluster_test(liq_measures_all_)
    state_probs_ = cluster_state(scaled_liq_, nstates=3)
    print(f'State probabilities are {state_probs_.mean(axis=0)}')
    pca_test(scaled_liq_)
    state_probs__, pca_ = gmm_pca(scaled_liq_)
    print(f'State probabilities are {state_probs__.mean(axis=0)}')
    loadings = pca_.components_.T * np.sqrt(pca_.explained_variance_ratio_)
    loading_matrix = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=liq_measures_all_.columns)
    print(loading_matrix)
    gmcm_test(scaled_liq_)

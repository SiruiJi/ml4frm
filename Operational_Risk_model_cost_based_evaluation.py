import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, f1_score)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import sys

sys.modules['sklearn.externals.joblib'] = joblib
from costcla.metrics import cost_loss, savings_score
from costcla.models import BayesMinimumRiskClassifier
from costcla.models import CostSensitiveLogisticRegression
from costcla.models import CostSensitiveDecisionTreeClassifier
from costcla.models import CostSensitiveRandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


def load_raw_data(file_path):
    fraud_data = pd.read_csv(file_path)
    del fraud_data['Unnamed: 0']

    fraud_data['time'] = pd.to_datetime(fraud_data['trans_date_trans_time'])
    del fraud_data['trans_date_trans_time']

    fraud_data['days'] = fraud_data['time'].dt.day_name()
    fraud_data['hour'] = fraud_data['time'].dt.hour

    return fraud_data


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km

    # Converting from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance


def data_preparation(fraud_data):
    fraud_data['dob'] = pd.to_datetime(fraud_data['dob'])
    final_date = dt.datetime(2023, 1, 1)
    fraud_data['age'] = ((final_date - fraud_data['dob']) / np.timedelta64(1, 'Y')).astype(int)

    fraud_data['distance'] = haversine_distance(fraud_data['lat'], fraud_data['long'], fraud_data['merch_lat'],
                                                fraud_data['merch_long'])

    numerical_fraud = fraud_data.select_dtypes(include=[np.number])
    del numerical_fraud['cc_num']
    del numerical_fraud['zip']
    del numerical_fraud['lat']
    del numerical_fraud['long']
    del numerical_fraud['unix_time']
    del numerical_fraud['merch_lat']
    del numerical_fraud['merch_long']
    del numerical_fraud['hour']
    del numerical_fraud['is_fraud']

    non_numerical_fraud = fraud_data.select_dtypes(include=['object']).copy()
    non_numerical_fraud['hour'] = fraud_data['hour']
    del non_numerical_fraud['merchant']
    del non_numerical_fraud['category']
    del non_numerical_fraud['first']
    del non_numerical_fraud['last']
    del non_numerical_fraud['street']
    del non_numerical_fraud['city']
    del non_numerical_fraud['job']
    del non_numerical_fraud['trans_num']

    dummies_fraud = pd.get_dummies(non_numerical_fraud, drop_first=True)
    dummies_fraud = dummies_fraud.astype(int)
    hour_dummies = pd.get_dummies(non_numerical_fraud['hour'], prefix='hour', drop_first=True).astype(int)
    dummies_fraud = pd.concat([dummies_fraud, hour_dummies], axis=1)
    del dummies_fraud['hour']  # Remove the original 'hour' column if needed

    fraud_df = pd.concat([numerical_fraud, dummies_fraud], axis=1)
    fraud_df['is_fraud'] = fraud_data['is_fraud']

    return numerical_fraud, non_numerical_fraud, dummies_fraud, fraud_df


def data_under_sampling(fraud_df):
    non_fraud_class = fraud_df[fraud_df['is_fraud'] == 0]
    fraud_class = fraud_df[fraud_df['is_fraud'] == 1]

    print('The number of observations in non_fraud_class is {}'.format(non_fraud_class['is_fraud'].value_counts()))
    print('The number of observation in fraud_class is {}'.format(fraud_class['is_fraud'].value_counts()))

    non_fraud_count, fraud_count = fraud_df['is_fraud'].value_counts()
    non_fraud_under = non_fraud_class.sample(fraud_count)

    under_sampled = pd.concat([non_fraud_under, fraud_class], axis=0)
    return non_fraud_class, fraud_class, non_fraud_under, under_sampled


def data_split(fraud_data):
    X = fraud_data.drop('is_fraud', axis=1)
    y = fraud_data['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test


def logistic_model_train_and_evaluate(X_train, X_test, y_train, y_test):
    param_log = {'C': np.logspace(-4, 4, 4), 'penalty': ['l1', 'l2']}
    log_grid = GridSearchCV(LogisticRegression(), param_grid=param_log, n_jobs=1)
    log_grid.fit(X_train, y_train)
    prediction_log = log_grid.predict(X_test)

    conf_mat_log = confusion_matrix(y_true=y_test, y_pred=prediction_log)

    print('Confusion matrix:\n', conf_mat_log)
    print('--' * 25)
    print('Classification report:\n', classification_report(y_test, prediction_log))
    return log_grid, conf_mat_log


def decision_tree_model_train_and_evaluate(X_train, X_test, y_train, y_test):
    param_dt = {'max_depth': [3, 5, 10],
                'min_samples_split': [2, 4, 6],
                'criterion': ['gini', 'entropy']}
    dt_grid = GridSearchCV(DecisionTreeClassifier(),
                           param_grid=param_dt, n_jobs=1)
    dt_grid.fit(X_train, y_train)
    prediction_dt = dt_grid.predict(X_test)

    conf_mat_dt = confusion_matrix(y_true=y_test, y_pred=prediction_dt)
    print('Confusion matrix:\n', conf_mat_dt)
    print('--' * 25)
    print('Classification report:\n', classification_report(y_test, prediction_dt))
    return dt_grid, conf_mat_dt


def random_forest_model_train_and_evaluate(X_train, X_test, y_train, y_test):
    param_rf = {'n_estimators': [20, 50, 100],
                'max_depth': [3, 5, 10],
                'min_samples_split': [2, 4, 6],
                'max_features': ['auto', 'sqrt', 'log2']}
    rf_grid = GridSearchCV(RandomForestClassifier(),
                           param_grid=param_rf, n_jobs=1)
    rf_grid.fit(X_train, y_train)
    prediction_rf = rf_grid.predict(X_test)

    conf_mat_rf = confusion_matrix(y_true=y_test, y_pred=prediction_rf)

    print('Confusion matrix:\n', conf_mat_rf)
    print('--' * 25)
    print('Classification report:\n', classification_report(y_test, prediction_rf))
    return rf_grid, conf_mat_rf


def xgboost_model_train_and_evaluate(X_train, X_test, y_train, y_test):
    param_boost = {'learning_rate': [0.01, 0.1],
                   'max_depth': [3, 5, 7],
                   'subsample': [0.5, 0.7],
                   'colsample_bytree': [0.5, 0.7],
                   'n_estimators': [10, 20, 30]}
    boost_grid = GridSearchCV(XGBClassifier(),
                              param_grid=param_boost, n_jobs=1)
    boost_grid.fit(X_train, y_train)
    prediction_boost = boost_grid.predict(X_test)

    conf_mat_boost = confusion_matrix(y_true=y_test, y_pred=prediction_boost)

    print('Confusion matrix:\n', conf_mat_boost)
    print('--' * 25)
    print('Classification report:\n', classification_report(y_test, prediction_boost))
    return boost_grid, conf_mat_boost


def cost_based_fraud_examination(fraud_df, conf_mat_log, conf_mat_dt, conf_mat_rf, conf_mat_boost):
    fraud_df_sampled = fraud_df.sample(int(len(fraud_df) * 0.2))
    cost_fp = 2
    cost_fn = fraud_df_sampled['amt']
    cost_tp = 2
    cost_tn = 0
    cost_mat = np.array([cost_fp * np.ones(fraud_df_sampled.shape[0]),
                         cost_fn,
                         cost_tp * np.ones(fraud_df_sampled.shape[0]),
                         cost_tn * np.ones(fraud_df_sampled.shape[0])]).T

    cost_log = conf_mat_log[0][1] * cost_fp + conf_mat_log[1][0] * cost_fn.mean() + conf_mat_log[1][1] * cost_tp
    cost_dt = conf_mat_dt[0][1] * cost_fp + conf_mat_dt[1][0] * cost_fn.mean() + conf_mat_dt[1][1] * cost_tp
    cost_rf = conf_mat_rf[0][1] * cost_fp + conf_mat_rf[1][0] * cost_fn.mean() + conf_mat_rf[1][1] * cost_tp
    cost_boost = conf_mat_boost[0][1] * cost_fp + conf_mat_boost[1][0] * cost_fn.mean() + conf_mat_boost[1][1] * cost_tp

    cost_df = pd.DataFrame()
    cost_df['model'] = ['lg', 'dt', 'rf', 'boost']
    cost_df['cost'] = [int(cost_log), int(cost_dt), int(cost_rf), int(cost_boost)]
    print(cost_df)

    X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = train_test_split(
        fraud_df_sampled.drop('is_fraud', axis=1), fraud_df_sampled.is_fraud, cost_mat, test_size=0.2, random_state=0)

    saving_models = []
    saving_models.append(('Log.Reg.', LogisticRegression()))
    saving_models.append(('Dec. Tree', DecisionTreeClassifier()))
    saving_models.append(('Random Forest', RandomForestClassifier()))

    savings_score_base_all = []
    for name, save_model in saving_models:
        sv_model = save_model
        sv_model.fit(X_train, y_train)
        y_pred = sv_model.predict(X_test)
        savings_score_base = savings_score(y_test, y_pred, cost_mat_test)
        savings_score_base_all.append(savings_score_base)
        print('The saving score for {} is {:.4f}'.format(name, savings_score_base))
        print('--' * 20)

    f1_score_base_all = []
    for name, save_model in saving_models:
        sv_model = save_model
        sv_model.fit(X_train, y_train)
        y_pred = sv_model.predict(X_test)
        f1_score_base = f1_score(y_test, y_pred)
        f1_score_base_all.append(f1_score_base)
        print('The F1 score for {} is {:.4f}'.format(name, f1_score_base))
        print('--' * 20)

    cost_sen_models = []
    cost_sen_models.append(('Log.Reg. CS', CostSensitiveLogisticRegression()))
    cost_sen_models.append(('Dec. Tree CS', CostSensitiveDecisionTreeClassifier()))
    cost_sen_models.append(('Random Forest CS', CostSensitiveRandomForestClassifier()))

    saving_cost_all = []
    for name, cost_model in cost_sen_models:
        cost_model.fit(np.array(X_train), np.array(y_train), cost_mat_train)
        y_pred = cost_model.predict(np.array(X_test))
        savings_score_cost = savings_score(np.array(y_test), np.array(y_pred), cost_mat_test)
        saving_cost_all.append(savings_score_cost)
        print('The saving score for {} is {:.4f}'.format(name, savings_score_cost))
        print('--' * 20)

    f1_score_cost_all = []
    for name, cost_model in cost_sen_models:
        cost_model.fit(np.array(X_train), np.array(y_train), cost_mat_train)
        y_pred = cost_model.predict(np.array(X_test))
        f1_score_cost = f1_score(np.array(y_test), np.array(y_pred))
        f1_score_cost_all.append(f1_score_cost)
        print('The F1 score for {} is {:.4f}'.format(name, f1_score_cost))
        print('--' * 20)

    saving_score_bmr_all = []
    for name, bmr_model in saving_models:
        f = bmr_model.fit(X_train, y_train)
        y_prob_test = f.predict_proba(np.array(X_test))
        f_bmr = BayesMinimumRiskClassifier()
        f_bmr.fit(np.array(y_test), y_prob_test)
        y_pred_test = f_bmr.predict(np.array(y_prob_test), cost_mat_test)
        saving_score_bmr = savings_score(y_test, y_pred_test,cost_mat_test)
        saving_score_bmr_all.append(saving_score_bmr)
        print('The saving score for {} is {:.4f}'.format(name, saving_score_bmr))
        print('--' * 20)

    f1_score_bmr_all = []
    for name, bmr_model in saving_models:
        f = bmr_model.fit(X_train, y_train)
        y_prob_test = f.predict_proba(np.array(X_test))
        f_bmr = BayesMinimumRiskClassifier()
        f_bmr.fit(np.array(y_test), y_prob_test)
        y_pred_test = f_bmr.predict(np.array(y_prob_test), cost_mat_test)
        f1_score_bmr = f1_score(y_test, y_pred_test)
        f1_score_bmr_all.append(f1_score_bmr)
        print('The F1 score for {} is {:.4f}'.format(name, f1_score_bmr))
        print('--' * 20)

    savings = [savings_score_base_all, saving_cost_all, saving_score_bmr_all]
    f1 = [f1_score_base_all, f1_score_cost_all, f1_score_bmr_all]
    saving_scores = pd.concat([pd.Series(x) for x in savings])
    f1_scores = pd.concat(pd.Series(x) for x in f1)
    scores = pd.concat([saving_scores, f1_scores], axis=1)
    scores.columns = ['saving_scores', 'F1_scores']

    model_names = ['Log. Reg_base', 'Dec. Tree_base', 'Random Forest_base', 'Log. Reg_cs', 'Dec. Tree_cs',
                   'Random Forest_cs', 'Log. Reg_bayes', 'Dec. Tree_bayes', 'Random Forest_bayes']
    plt.figure(figsize=(10, 6))
    plt.plot(range(scores.shape[0]), scores['F1_scores'], '--', label='F1Score')
    plt.bar(np.arange(scores.shape[0]), scores['saving_scores'], 0.6, label='Savings')
    _ = np.arange(len(model_names))
    plt.xticks(_, model_names)
    plt.legend(loc='best')
    plt.xticks(rotation='vertical')
    plt.show()


if __name__ == '__main__':
    file_path = 'D:/PyCharm Community Edition 2023.1.2/Python_Project/Finance/ml4frm/fraudTrain.csv'
    fraud_data_ = load_raw_data(file_path)
    numerical_fraud_, non_numerical_fraud_, dummies_fraud_, fraud_df_ = data_preparation(fraud_data_)
    non_fraud_class_, fraud_class_, non_fraud_under_, under_sampled_ = data_under_sampling(fraud_df_)
    X_train_, X_test_, y_train_, y_test_ = data_split(under_sampled_)
    log_model, conf_mat_log_ = logistic_model_train_and_evaluate(X_train_, X_test_, y_train_, y_test_)
    dt_model, conf_mat_dt_ = decision_tree_model_train_and_evaluate(X_train_, X_test_, y_train_, y_test_)
    rf_model, conf_mat_rf_ = random_forest_model_train_and_evaluate(X_train_, X_test_, y_train_, y_test_)
    xgb_model, conf_mat_boost_ = xgboost_model_train_and_evaluate(X_train_, X_test_, y_train_, y_test_)
    cost_based_fraud_examination(fraud_df_, conf_mat_log_, conf_mat_dt_, conf_mat_rf_, conf_mat_boost_)

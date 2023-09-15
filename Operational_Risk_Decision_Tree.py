import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, f1_score)


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

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    distance = R * c
    return distance


def data_preparation(fraud_data):
    fraud_data['dob'] = pd.to_datetime(fraud_data['dob'])
    final_date = dt.datetime(2023, 1, 1)
    fraud_data['age'] = ((final_date - fraud_data['dob']) / np.timedelta64(1, 'Y')).astype(int)

    fraud_data['distance'] = haversine_distance(fraud_data['lat'], fraud_data['long'], fraud_data['merch_lat'], fraud_data['merch_long'])

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


def decision_tree_model_train_and_evaluate(X_train, X_test, y_train, y_test):
    param_dt = {'max_depth': [3, 5, 10],
                'min_samples_split': [2, 4, 6],
                'criterion': ['gini', 'entropy']}
    dt_grid = GridSearchCV(DecisionTreeClassifier(),
                           param_grid=param_dt, n_jobs=-1)
    dt_grid.fit(X_train, y_train)
    prediction_dt = dt_grid.predict(X_test)

    conf_mat_dt = confusion_matrix(y_true=y_test, y_pred= prediction_dt)
    print('Confusion matrix:\n', conf_mat_dt)
    print('--' * 25)
    print('Classification report:\n', classification_report(y_test, prediction_dt))
    return dt_grid


if __name__ == '__main__':
    file_path = 'D:/PyCharm Community Edition 2023.1.2/Python_Project/Finance/ml4frm/fraudTrain.csv'
    fraud_data_ = load_raw_data(file_path)
    numerical_fraud_, non_numerical_fraud_, dummies_fraud_, fraud_df_ = data_preparation(fraud_data_)
    non_fraud_class_, fraud_class_, non_fraud_under_, under_sampled_ = data_under_sampling(fraud_df_)
    X_train_, X_test_, y_train_, y_test_ = data_split(under_sampled_)
    dt_model = decision_tree_model_train_and_evaluate(X_train_, X_test_, y_train_, y_test_)
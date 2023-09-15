import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom
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

    standard = StandardScaler()
    scaled_numerical_fraud = standard.fit_transform(numerical_fraud)
    scaled_numerical_fraud_df = pd.DataFrame(scaled_numerical_fraud, columns=numerical_fraud.columns)
    scaled_fraud_df = scaled_numerical_fraud_df
    scaled_fraud_df['is_fraud'] = fraud_data['is_fraud']

    return numerical_fraud, non_numerical_fraud, dummies_fraud, fraud_df, scaled_fraud_df


def data_under_sampling(fraud_df):
    non_fraud_class = fraud_df[fraud_df['is_fraud'] == 0]
    fraud_class = fraud_df[fraud_df['is_fraud'] == 1]

    print('The number of observations in non_fraud_class is {}'.format(non_fraud_class['is_fraud'].value_counts()))
    print('The number of observation in fraud_class is {}'.format(fraud_class['is_fraud'].value_counts()))

    non_fraud_count, fraud_count = fraud_df['is_fraud'].value_counts()
    non_fraud_under = non_fraud_class.sample(fraud_count)

    under_sampled = pd.concat([non_fraud_under, fraud_class], axis=0)
    return non_fraud_class, fraud_class, non_fraud_under, under_sampled


def self_organized_mapping(under_sampled):
    X = under_sampled.drop('is_fraud', axis=1).to_numpy()  # Converting DataFrame to NumPy array
    y = under_sampled['is_fraud']

    som = MiniSom(2, 1, X.shape[1], sigma=0.5, learning_rate=0.5)
    som.train_random(X, num_iteration=1000)  # Added num_iteration

    predictions_som = [som.winner(x) for x in X]
    predictions_som_1D = [x[0] * 1 + x[1] for x in predictions_som]

    print('Classification report: \n',
          classification_report(y, predictions_som_1D))

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))
    x_axis = X[:, 0]
    y_axis = X[:, 1]
    ax[0].scatter(x_axis, y_axis, alpha=0.1, cmap='Greys', c=y)
    ax[0].title.set_text('Actual Classes')
    ax[1].scatter(x_axis, y_axis, alpha=0.1, cmap='Greys', c=predictions_som_1D)
    ax[1].title.set_text('SOM Predictions')
    plt.show()


if __name__ == '__main__':
    file_path = 'D:/PyCharm Community Edition 2023.1.2/Python_Project/Finance/ml4frm/fraudTrain.csv'
    fraud_data_ = load_raw_data(file_path)
    numerical_fraud_, non_numerical_fraud_, dummies_fraud_, fraud_df_, scaled_fraud_df_ = data_preparation(fraud_data_)
    non_fraud_class_, fraud_class_, non_fraud_under_, under_sampled_ = data_under_sampling(scaled_fraud_df_)
    self_organized_mapping(under_sampled_)
    # useless model

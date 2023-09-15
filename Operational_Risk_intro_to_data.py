import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import seaborn as sns


def load_raw_data(file_path):
    fraud_data = pd.read_csv(file_path)
    del fraud_data['Unnamed: 0']

    fraud_data['time'] = pd.to_datetime(fraud_data['trans_date_trans_time'])
    del fraud_data['trans_date_trans_time']

    fraud_data['days'] = fraud_data['time'].dt.day_name()
    fraud_data['hour'] = fraud_data['time'].dt.hour

    return fraud_data


def data_info(fraud_data):
    print(fraud_data.info())
    plt.pie(fraud_data['is_fraud'].value_counts(), labels=[0, 1])
    plt.title('Pie Chart for Dependent Variable')
    print(fraud_data['is_fraud'].value_counts())
    plt.show()

    null_values = fraud_data.isnull().sum()
    plt.barh(null_values.index, null_values.values)
    plt.xlabel('Number of Missing Values')
    plt.ylabel('Columns')
    plt.title('Missing Values per Column')
    plt.show()


def fraud_cat(cols, fraud_data):
    k = 1
    plt.figure(figsize=(20, 40))
    for i in cols:
        categ = fraud_data.loc[fraud_data['is_fraud'] == 1, i].value_counts().sort_values(ascending=False).head(10)
        plt.subplot(int(len(cols)//2), int(len(cols) // 2), k)
        bar_plot = plt.bar(categ.index, categ.values)
        plt.title(f'Top 10 Fraud Cases per {i} Categories')
        plt.xticks(rotation=45)
        k += 1
    plt.show()


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


def data_corr(numerical_fraud):
    plt.figure(figsize=(10, 6))
    corr_mat = numerical_fraud.corr()
    sns.heatmap(corr_mat, annot=True, cmap='viridis')
    plt.show()


if __name__ == '__main__':
    file_path = 'D:/PyCharm Community Edition 2023.1.2/Python_Project/Finance/ml4frm/fraudTrain.csv'
    fraud_data_ = load_raw_data(file_path)
    data_info(fraud_data_)
    cols = ['job', 'state', 'gender', 'category', 'days', 'hour']
    fraud_cat(cols, fraud_data_)
    numerical_fraud_, non_numerical_fraud_, dummies_fraud_, fraud_df_ = data_preparation(fraud_data_)
    data_corr(numerical_fraud_)


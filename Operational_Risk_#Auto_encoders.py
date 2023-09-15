from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras.layers import Dense, Dropout
from keras import regularizers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split


def load_raw_data(file_path):
    fraud_data = pd.read_csv(file_path)
    del fraud_data['Unnamed: 0']

    fraud_data['time'] = pd.to_datetime(fraud_data['trans_date_trans_time'])
    del fraud_data['trans_date_trans_time']

    fraud_data['days'] = fraud_data['time'].dt.day_name()
    fraud_data['hour'] = fraud_data['time'].dt.hour

    scaled_values = StandardScaler().fit_transform(fraud_data[['amt', 'city_pop', 'hour']])
    fraud_df = pd.DataFrame(scaled_values, columns=['amt', 'city_pop', 'hour'])
    fraud_df['is_fraud'] = fraud_data['is_fraud']


    return fraud_data, fraud_df


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


def auto_encoders(X_train, X_test, y_train, y_test):
    autoencoder = keras.Sequential()
    autoencoder.add(Dense(X_train.shape[1], activation='tanh',
                          activity_regularizer=regularizers.l1(10e-5),
                          input_dim=X_train.shape[1]))

    autoencoder.add(Dense(64, activation='tanh'))
    autoencoder.add(Dense(32, activation='relu'))

    autoencoder.add(Dense(32, activation='elu'))
    autoencoder.add(Dense(64, activation='tanh'))
    autoencoder.add(Dense(X_train.shape[1], activation='elu'))
    autoencoder.compile(loss='mse', optimizer='adam')
    print(autoencoder.summary())

    batch_size = 200
    epochs = 100
    history = autoencoder.fit(X_train, X_train,
                              shuffle=True, epochs=epochs, batch_size=batch_size,
                              validation_data=(X_test, X_test), verbose=0).history
    autoencoder_pred = autoencoder.predict(X_test)
    mse = np.mean(np.power(X_test - autoencoder_pred, 2), axis=1)
    error_df = pd.DataFrame({'reconstruction_error': mse, 'true_class': y_test})
    print(error_df.describe())

    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], linewidth=2, label='Train')
    plt.plot(history['val_loss'], linewidth=2, label='Test')
    plt.legend(loc='upper right')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()


if __name__ == '__main__':
    file_path = 'D:/PyCharm Community Edition 2023.1.2/Python_Project/Finance/ml4frm/fraudTrain.csv'
    fraud_data_, fraud_df_ = load_raw_data(file_path)
    non_fraud_class_, fraud_class_, non_fraud_under_, under_sampled_ = data_under_sampling(fraud_df_)
    X_train_, X_test_, y_train_, y_test_ = data_split(under_sampled_)
    auto_encoders(X_train_, X_test_, y_train_, y_test_)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

def read_original_files(file_path):
    credit = pd.read_csv(file_path)
    print(credit.head())
    del credit['Unnamed: 0']
    return credit


def data_conversion(credit):
    print(credit.describe())
    numerical_credit = credit.select_dtypes(include=[np.number])
    '''obtain all numerical variables'''
    plt.figure(figsize=(10, 8))
    k = 0
    cols = numerical_credit.columns
    for i, j in zip(range(len(cols)), cols):
        k += 1
        plt.subplot(2, 2, k)
        plt.hist(numerical_credit.iloc[:, i])
        plt.title(j)
    plt.show()

    scaler = StandardScaler()
    scaled_credit = scaler.fit_transform(numerical_credit)
    scaled_credit = pd.DataFrame(scaled_credit, columns=numerical_credit.columns)

    non_numerical_credit = credit.select_dtypes(include=['object'])
    dummies_credit = pd.get_dummies(non_numerical_credit, drop_first=True)
    dummies_credit = dummies_credit.astype(int)
    print(dummies_credit.head())

    combined_credit = pd.concat([scaled_credit, dummies_credit], axis=1)

    return numerical_credit, scaled_credit, dummies_credit, combined_credit


def data_preparation(combined_credit):
    X = combined_credit.drop("Risk_good", axis=1)
    y = combined_credit["Risk_good"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def DL_risk(dropout_rate, verbose=0):
    model = keras.Sequential()
    model.add(Dense(128,kernel_initializer='normal', activation='relu', input_dim=21))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return model


def training_model(X_train, X_test, y_train, y_test):
    parameters = {'batch_size': [10, 50, 100],
                  'epochs': [50, 100, 150],
                  'dropout_rate': [0.2, 0.4]}
    model = KerasClassifier(build_fn=DL_risk)
    gs = GridSearchCV(estimator=model, param_grid=parameters, scoring='roc_auc', error_score='raise')

    gs.fit(X_train, y_train, verbose=0)
    print('Best hyperparameters for first cluster in DL are {}'.format(gs.best_params_))

    model = KerasClassifier(build_fn=DL_risk,
                            dropout_rate=gs.best_params_['dropout_rate'],
                            verbose=0,
                            batch_size=gs.best_params_['batch_size'],
                            epochs=gs.best_params_['epochs'])
    model.fit(X_train, y_train)
    DL_predict = model.predict(X_test)
    DL_ROC_AUC = roc_auc_score(y_test, pd.DataFrame(DL_predict.flatten()))
    print('DL_ROC_AUC is {:.4f}'.format(DL_ROC_AUC))
    return model, DL_predict


if __name__ == '__main__':
    file_path = 'D:/PyCharm Community Edition 2023.1.2/Python_Project/Finance/py4frm/german_credit_data.csv'
    credit_ = read_original_files(file_path)
    numerical_credit_, scaled_credit_, dummies_credit_, combined_credit_ = data_conversion(credit_)
    X_train_, X_test_, y_train_, y_test_ = data_preparation(combined_credit_)
    model_, DL_pred = training_model( X_train_, X_test_, y_train_, y_test_)

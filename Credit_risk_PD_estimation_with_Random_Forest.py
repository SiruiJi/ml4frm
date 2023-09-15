import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.ensemble import RandomForestClassifier


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


def training_model(X_train, X_test, y_train, y_test):
    rfc = RandomForestClassifier(random_state=42)

    param_rfc = {'n_estimators': [100, 300],
                 'criterion': ['gini', 'entropy'],
                 'max_features': ['auto', 'sqrt', 'log2'],
                 'max_depth': [3, 4, 5, 6],
                 'min_samples_split': [5, 10]}

    halve_RF = HalvingRandomSearchCV(rfc, param_rfc, scoring='roc_auc', n_jobs=-1)
    halve_RF.fit(X_train, y_train)
    y_pred_SVC = halve_RF.predict(X_test)
    print('The ROC AUC score of RF is {:4f}'.format(roc_auc_score(y_test, y_pred_SVC)))


    return halve_RF, y_pred_SVC


if __name__ == '__main__':
    file_path = 'D:/PyCharm Community Edition 2023.1.2/Python_Project/Finance/py4frm/german_credit_data.csv'
    credit_ = read_original_files(file_path)
    numerical_credit_, scaled_credit_, dummies_credit_, combined_credit_ = data_conversion(credit_)
    X_train_, X_test_, y_train_, y_test_ = data_preparation(combined_credit_)
    model, y_pred = training_model(X_train_, X_test_, y_train_, y_test_)
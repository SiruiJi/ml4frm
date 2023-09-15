import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import roc_auc_score, roc_curve


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
    param_NN ={"hidden_layer_sizes": [(100, 50), (50, 50), (10, 100)],
               "solver": ["lbfgs", "sgd", "adam"],
               "learning_rate_init": [0.001, 0.05]}
    MLP = MLPClassifier(random_state=42)

    param_halve_NN = HalvingRandomSearchCV(MLP, param_NN, scoring='roc_auc')
    param_halve_NN.fit(X_train, y_train)

    y_pred_NN = param_halve_NN.predict(X_test)
    print('The ROC AUC score of RF is {:4f}'.format(roc_auc_score(y_test, y_pred_NN)))

    return param_halve_NN, y_pred_NN


if __name__ == '__main__':
    file_path = 'D:/PyCharm Community Edition 2023.1.2/Python_Project/Finance/py4frm/german_credit_data.csv'
    credit_ = read_original_files(file_path)
    numerical_credit_, scaled_credit_, dummies_credit_, combined_credit_ = data_conversion(credit_)
    X_train_, X_test_, y_train_, y_test_ = data_preparation(combined_credit_)
    model, y_pred = training_model(X_train_, X_test_, y_train_, y_test_)
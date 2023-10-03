import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve

sns.set()
plt.rcParams["figure.figsize"] = (10, 6)


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
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train)
    coefficients = logistic_regression.coef_
    intercept = logistic_regression.intercept_
    coef_df = pd.DataFrame(data=coefficients, columns=X_train.columns)
    print("Intercept: ", intercept[0])

    y_pred = logistic_regression.predict(X_test)
    y_score = logistic_regression.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    print('area under ROC curve is ', roc_auc)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return logistic_regression, intercept, coef_df


if __name__ == '__main__':
    file_path = 'german_credit_data.csv'
    credit_ = read_original_files(file_path)
    numerical_credit_, scaled_credit_, dummies_credit_, combined_credit_ = data_conversion(credit_)
    X_train_, X_test_, y_train_, y_test_ = data_preparation(combined_credit_)
    model, intercept, coef= training_model(X_train_, X_test_, y_train_, y_test_)


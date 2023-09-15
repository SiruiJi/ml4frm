import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer, KElbowVisualizer


sns.set()
plt.rcParams["figure.figsize"] = (10, 6)


def read_original_files(file_path):
    credit = pd.read_csv(file_path)
    print(credit.head())
    del credit['Unnamed: 0']
    del credit['Risk']
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


def elbow_method(combined_credit):
    distance = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(combined_credit)
        distance.append(kmeans.inertia_)

    plt.plot(range(1, 10), distance, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method')
    plt.show()


def silhouette_score_method(combined_credit):
    fig, ax = plt.subplots(4, 2, figsize=(25, 20))
    for i in range(2, 10):
        km = KMeans(n_clusters=i)
        km.fit(combined_credit)
        labels = km.labels_
        sil_score = silhouette_score(combined_credit, labels, metric='euclidean')
        print(f"For n_clusters = {i}, the silhouette score is {sil_score}")
        q, r = divmod(i, 2)
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q - 1][r])
        visualizer.fit(combined_credit)
        ax[q - 1][r].set_title("For Cluster_"+str(i))
        ax[q - 1][r].set_xlabel("Silhouette Score")
    plt.show()


def ch_score(combined_credit):
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(2, 10),
                                  metric='calinski_harabasz',
                                  timings=False)
    visualizer.fit(combined_credit)
    visualizer.show()


if __name__ == '__main__':
    '''https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk?resource=download'''
    file_path = 'D:/PyCharm Community Edition 2023.1.2/Python_Project/Finance/py4frm/german_credit_data.csv'
    credit_ = read_original_files(file_path)
    numerical_credit_, scaled_credit_, dummies_credit_, combined_credit_ = data_preparation(credit_)
    elbow_method(combined_credit_)
    silhouette_score_method(combined_credit_)
    ch_score(combined_credit_)

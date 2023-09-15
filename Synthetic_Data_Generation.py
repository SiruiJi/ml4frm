from sklearn.datasets import make_regression, make_classification, make_blobs
import matplotlib.pyplot as plt


def generating_synthetic_data_from_regression():
    X, y = make_regression(n_samples=1000, n_features=3, noise=0.2, random_state=123)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.3, cmap='Greys', c=y)

    plt.figure(figsize=(18, 18))
    k = 0
    for i in range(0, 10):
        X, y = make_regression(n_samples=100, n_features=3, noise=i, random_state=123)
        k += 1
        plt.subplot(5, 2, k)
        plt.scatter(X[:, 0], X[:, 1], alpha=0.3, cmap='Greys', c=y)
        plt.title('Synthetic Data with Different Noises: ' + str(i))
    plt.show()


def generating_synthetic_data_from_classification():
    X, y = make_classification(n_samples=100, n_features=4, n_classes=7,
                               n_redundant=0, n_informative=4, random_state=123)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.3, cmap='Greys', c=y)

    plt.figure(figsize=(18, 18))
    k = 0
    for i in range(2, 6):
        X, y = make_classification(n_samples=100, n_features=4, n_classes=i,
                                   n_redundant=0, n_informative=4, random_state=123)
        k += 1
        plt.subplot(2, 2, k)
        plt.scatter(X[:, 0], X[:, 1], alpha=0.8, cmap='Greys', c=y)
        plt.title('Synthetic Data with Different Classes: ' + str(i))
    plt.show()


def generating_synthetic_data_from_clusters():
    X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=0)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.3, cmap='Greys', c=y)

    plt.figure(figsize=(18, 18))
    k = 0
    for i in range(2, 6):
        X, y = make_blobs(n_samples=100, centers=i, n_features=2, random_state=0)
        k += 1
        plt.subplot(2, 2, k)
        my_scatter_plot = plt.scatter(X[:, 0], X[:, 1], alpha=0.3, cmap='gray', c=y)
        plt.title('Synthetic Data with Different Clusters: ' + str(i))
    plt.show()


if __name__ == '__main__':
    generating_synthetic_data_from_regression()
    generating_synthetic_data_from_classification()
    generating_synthetic_data_from_clusters()

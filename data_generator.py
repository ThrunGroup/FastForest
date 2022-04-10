import numpy as np


def create_data(N=1000):
    """
    Creates some toy data. The label y is randomly chosen as 0 or 1 with equal probability. The second feature is
    randomly generated with no correlation with y. The first feature is a Gaussian centered on y.

    Datasets created by this method should be split by the first feature at 0.5.

    :param N: Dataset size
    :return: dataset
    """
    y = np.random.choice([0, 1], size=(N))
    X = np.zeros((N, 3))
    X[:, 2] = y
    X[:, 0] = np.random.normal(loc=y, scale=0.2, size=N)
    X[:, 1] = np.random.rand(N)
    return X
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from mnist import MNIST
from typing import List

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype

from utils.constants import FLIGHT, AIR, APS, BLOG, SKLEARN_REGRESSION, MNIST_STR, HOUSING, COVTYPE, KDD


def get_dummies(d, col):
    dd = pd.get_dummies(d[col])
    dd.columns = [col + "_%s" % c for c in dd.columns]
    return dd


def get_data(
    filename: str,
    vars_categ: List[str],
    vars_num: List[str],
    var_target: str,
    train_to_test: float = 0.9,
    seed: int = 0,
    is_flight: bool = False,
    is_aps: bool = False,
):
    # TODO(@motiwari): Fix this
    d_train_test = pd.read_csv(filename)

    # Fill nan values and shuffle
    d_train_test = d_train_test.replace(["na"], np.nan)
    d_train_test.fillna(method="bfill", inplace=True)
    d_train_test.fillna(method="ffill", inplace=True)
    d_train_test.sample(frac=1, random_state=seed)
    if len(vars_categ) > 0:
        X_train_test_categ = pd.concat(
            [get_dummies(d_train_test, col) for col in vars_categ], axis=1
        )
    else:
        X_train_test_categ = pd.DataFrame()
    X_train_test = pd.concat(
        [X_train_test_categ, d_train_test[vars_num]], axis=1
    ).to_numpy(dtype=np.float)
    if is_flight:
        y_train_test = np.where(d_train_test[var_target] == "Y", 1, 0)
    elif is_aps:
        y_train_test = np.where(d_train_test[var_target] == "pos", 1, 0)
    else:
        y_train_test = d_train_test[var_target].to_numpy().squeeze()

    train_size = int(d_train_test.shape[0] * train_to_test)
    X_train = X_train_test[0:train_size]
    y_train = y_train_test[0:train_size]
    X_test = X_train_test[train_size:]
    y_test = y_train_test[train_size:]
    return X_train, y_train, X_test, y_test


def get_air_data(train_to_test: float = 0.9, seed: int = 0):
    # Regression
    # Download from https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data
    # Merge multiple csv files to one
    filename = "air_data.csv"
    vars_categ = ["station", "WSPM"]
    vars_num = [
        "year",
        "month",
        "day",
        "hour",
        "PM2.5",
        "PM10",
        "SO2",
        "NO2",
        "TEMP",
        "PRES",
        "DEWP",
        "RAIN",
    ]
    var_target = "O3"
    return get_data(
        filename=filename,
        vars_categ=vars_categ,
        vars_num=vars_num,
        var_target=var_target,
        train_to_test=train_to_test,
        seed=seed,
        is_flight=False,
    )


def get_small_flight_data(train_to_test: float = 0.9, seed: int = 0):
    # Classification
    # Download from https://github.com/szilard/benchm-ml/tree/master/z-other-tools
    filename = "flight_0.1m_data.csv"
    vars_categ = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
    vars_num = ["DepTime", "Distance"]
    var_target = "dep_delayed_15min"
    return get_data(
        filename=filename,
        vars_categ=vars_categ,
        vars_num=vars_num,
        var_target=var_target,
        train_to_test=train_to_test,
        seed=seed,
        is_flight=True,
    )


def get_large_flight_data(train_to_test: float = 0.9, seed: int = 0):
    # Classificaiton
    # Download from https://github.com/szilard/benchm-ml/tree/master/z-other-tools
    filename = "flight_1m_data.csv"
    vars_categ = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
    vars_num = ["DepTime", "Distance"]
    var_target = "dep_delayed_15min"
    return get_data(
        filename=filename,
        vars_categ=vars_categ,
        vars_num=vars_num,
        var_target=var_target,
        train_to_test=train_to_test,
        seed=seed,
        is_flight=True,
    )


def get_aps_data(train_to_test: float = 0.9, seed: int = 0):
    # Classification
    # Download from https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks
    filename = "aps_data.csv"
    vars_categ = []
    vars_num = list(pd.read_csv(filename).columns)[1:]  # first column is target
    var_target = "class"
    return get_data(
        filename=filename,
        vars_categ=vars_categ,
        vars_num=vars_num,
        var_target=var_target,
        train_to_test=train_to_test,
        seed=seed,
        is_aps=True,
    )


def get_blog_data(train_to_test: float = 0.9, seed: int = 0):
    # Regression
    # Download from https://archive.ics.uci.edu/ml/datasets/BlogFeedback
    filename = "blog_data.csv"
    vars_categ = []
    vars_num = list(pd.read_csv(filename).columns)[:-1]  # last column is target
    var_target = "target"
    return get_data(
        filename=filename,
        vars_categ=vars_categ,
        vars_num=vars_num,
        var_target=var_target,
        train_to_test=train_to_test,
        seed=seed,
    )


def get_sklearn_data(data_size: int = 200000, n_features: int = 50, informative_ratio: float = 0.06, seed: int = 1, epsilon: float = 0.01, use_dynamic_eps: bool = False):
    # sklearn regression datasets
    params = {
        "data_size": data_size,
        "n_features": n_features,
        "informative_ratio": informative_ratio,
        "seed": seed,
        "epsilon": epsilon,
    }

    full_data, full_targets = sklearn.datasets.make_regression(
        params["data_size"],
        n_features=params["n_features"],
        n_informative=int(params["n_features"] * params["informative_ratio"]),
        random_state=params["seed"],
    )

    train_test_split = int(0.8 * params["data_size"])
    train_data = full_data[:train_test_split]
    train_targets = full_targets[:train_test_split]

    test_data = full_data[train_test_split:]
    test_targets = full_targets[train_test_split:]
    return train_data, train_targets, test_data, test_targets


def get_mnist():
    mndata = MNIST(os.path.join("..", "mnist"))

    train_images, train_labels = mndata.load_training()
    rng = np.random.default_rng(0)
    subsample_idcs = rng.choice(len(train_images), 4 * len(train_images))
    train_images = np.array(train_images)[subsample_idcs]
    train_labels = np.array(train_labels)[subsample_idcs]

    test_images, test_labels = mndata.load_testing()
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    return train_images, train_labels, test_images, test_labels


def get_housing():
    cal_housing = fetch_california_housing()
    X = cal_housing.data
    y = cal_housing.target
    y -= y.mean()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, y_train, X_test, y_test


def get_covtype():
    covtype = fetch_covtype()
    X = covtype.data
    y = covtype.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, y_train, X_test, y_test


def get_kdd():
    all_data = np.load("../datasets/kdd98.npz.npy", allow_pickle=True)
    rng = np.random.default_rng()
    rng.shuffle(all_data) # in-place shuffle

    TARGET_IDX = 471  # TODO(@motiwari): Ensure this isn't off-by-one
    y = all_data[:, TARGET_IDX]
    all_data = np.delete(all_data, TARGET_IDX, 1)
    all_data = pd.get_dummies(all_data) # Fix this

    X_train, X_test, y_train, y_test = train_test_split(all_data, y, test_size=0.2, random_state=0)
    import ipdb; ipdb.set_trace()
    return X_train, y_train, X_test, y_test


def fetch_data(dataset: str):
    if dataset is FLIGHT:
        return get_small_flight_data()
    elif dataset is AIR:
        return get_air_data()
    elif dataset is APS:
        return get_aps_data()
    elif dataset is BLOG:
        return get_blog_data()
    elif dataset is SKLEARN_REGRESSION:
        return get_sklearn_data()
    elif dataset is MNIST_STR:
        return get_mnist()
    elif dataset is HOUSING:
        return get_housing()
    elif dataset is COVTYPE:
        return get_covtype()
    elif dataset is KDD:
        return get_kdd()
    else:
        raise NotImplementedError(f"{dataset} is not implemented")

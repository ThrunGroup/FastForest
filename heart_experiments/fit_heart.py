import pandas as pd
import numpy as np
import os
import wandb
import random
import preprocess_heart
import argparse

from csv import DictWriter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from imblearn.under_sampling import RandomUnderSampler

from typing import (
    Tuple,
    Iterable
)

from forest import Forest


def check_dimension(X: np.ndarray, Y: np.ndarray) -> bool:
    """
    Check the dimension of X and Y

    :param X: input dataset
    :param Y: target dataset
    :return: return whether the dimensions of X and Y are valid.
    """
    return (len(X) == len(Y)) and (len(X.shape) == 2) and (len(Y.shape) == 1)


def get_subset(
        X: np.ndarray, Y: np.ndarray, subset_size: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a subset of dataset X and Y.

    :param X: an input dataset
    :param Y: a target dataset
    :param subset_size: subset_size (# of rows) of X and Y
    :param seed: random seed when taking a subset of X and Y 
    :return: the subset of X and Y
    """
    assert len(Y) >= subset_size, "Invalid subset_size of dataframe"
    assert check_dimension(X, Y), "Invalid dimension size of dataset"

    random_state = np.random.RandomState(seed)
    idcs = random_state.choice(len(Y), subset_size, replace=False)
    return X[idcs, :], Y[idcs]


def split_data(
        X: np.ndarray,
        Y: np.ndarray,
        ratio: Iterable,
        subset_size: int = -1,
        seed: int = 1,
        is_balanced: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Take a subset of dataset and split it into train, validation, and test set

    :param X: an input dataset
    :param Y: a target dataset
    :param ratio: the ratio of splitting dataset into train, validation, and test set
    :param subset_size: subset_size (# of rows) of X and Y we want to consider
    :param seed: random seed
    :param is_balanced: whether to return balanced data
    :return: tuple of X_train, X_val, X_test, Y_train, Y_val, Y_test
    """
    if is_balanced:
        rus = RandomUnderSampler(random_state=seed)
        X, Y = rus.fit_sample(X, Y)

    ratio = np.array(ratio)
    assert len(ratio) == 3, "invalid split ratio of dataset"
    ratio /= np.sum(ratio)

    if subset_size != -1:
        X, Y = get_subset(X, Y, subset_size, seed)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=ratio[2], random_state=seed
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=ratio[1] / (ratio[0] + ratio[1]), random_state=seed
    )

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def load_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param filepath: filepath of csv file
    :return: Input data and target data
    """
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1].to_numpy()
    Y = df.iloc[:, -1].to_numpy()
    return X, Y


def append_dict_as_row(
        file_name: str, dict_of_elem: dict, field_names: Iterable
) -> None:
    """
    Taken from https://thispointer.com/python-how-to-append-a-new-row-to-an-existing-csv-file/
    Append "dict_of_elem" in csv file with "file_name". The columns of csv file are expected to be equal to
    "field_names".
    Ex) "hi.csv":  "a" "b" "c"
                    0   1   3
        append_dict_as row("hi.csv", ["a": 3, "b": 2, "c": 2"], ["a", "b", "c"]) gives
        "hi.csv":  "a" "b" "c"
                    0   1   3
                    3   2   2

    :param file_name: filename of csv file we want to append
    :param dict_of_elem: a dictionary that we want to append
    :param field_names: the keys of dictionary
    """
    with open(file_name, "a+", newline="") as write_obj:
        dict_writer = DictWriter(write_obj, fieldnames=field_names)
        dict_writer.writerow(dict_of_elem)


def fit(
        config: argparse.Namespace = None, use_wandb: bool = True, use_sweep: bool = True
) -> None:
    """
    Fit a random forest with config.algorithm to config.dataset with different hyperparameters specified in
    config variable and log the result

    :param config: contains values needed to construct random forest and train it.
    :param use_wandb: whether to use wandb api or not
    :param use_sweep: whether to use the sweep method of wandb api or not
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if use_wandb:
        wandb.init()
        # If called by wandb.agent, as below,
        # this config will be set by the sweep controller
        if use_sweep:
            config = wandb.config
        else:
            wandb.config = config

    np.random.seed(config.seed)
    random.seed(config.seed)

    if config.dataset == "HEART":
        filepath = os.path.join("../dataset", "new_heart_2020_cleaned.csv")
    else:
        raise NotImplementedError("Invalid choice of dataset")
    if not os.path.exists(filepath):
        preprocess_heart.main()

    X, Y = load_data(filepath)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = \
        split_data(X, Y, [0.6, 0.2, 0.2], config["sub_size"], config["seed"], config["is_balanced"])

    if config.algorithm == "SKLEARN":
        forest = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            bootstrap=config.bootstrap,
        )
        forest.fit(X_train, Y_train)
        val_acc = balanced_accuracy_score(Y_val, forest.predict(X_val))
        test_acc = balanced_accuracy_score(Y_test, forest.predict(X_test))
    elif config.algorithm == "FASTFOREST":
        forest = Forest(
            X_train,
            Y_train,
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            bootstrap=config.bootstrap,
        )
        forest.fit()
        val_acc = balanced_accuracy_score(Y_val, forest.predict_batch(X_val)[0])
        test_acc = balanced_accuracy_score(Y_test, forest.predict_batch(X_test)[0])
    else:
        raise NotImplementedError("Invalid choice of RF algorithm")

    if config.verbose:
        print(f"{config.algorithm}'s random forest balanced validation Accuracy:", val_acc)
        print(f"{config.algorithm}'s random forest balanced test Accuracy:", test_acc)

    log_dict = {"val_acc": val_acc, "test_acc": test_acc}
    if use_wandb:
        wandb.log(log_dict)
    log_dict.update(
        {
            "algorithm": config.algorithm,
            "bootstrap": config.bootstrap,
            "dataset": config.dataset,
            "sub_size": config.sub_size,
            "seed": config.seed,
            "max_depth": config.max_depth,
            "n_estimators": config.n_estimators,
            "verbose": config.verbose,
            "is_balanced": config.is_balanced
        }
    )
    log_filename = os.path.join("../hyperparams_sweep", "sweep_log.csv")
    if not os.path.exists(log_filename):
        os.makedirs("../hyperparams_sweep", exist_ok=True)
        df = pd.DataFrame(columns=log_dict.keys())
        df.to_csv(log_filename, index=False)
    append_dict_as_row(log_filename, log_dict, log_dict.keys())

import pandas as pd
import numpy as np
import os
import wandb
import random
import preprocess_heart
import argparse

from csv import DictWriter
from sklearn.model_selection import train_test_split
from forest import Forest
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Iterable
from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree,
    export_text,
)


def get_subset(
        X: np.ndarray, Y: np.ndarray, subset_size: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a subset of dataset X and Y.
    :param X: an array which has 2 dimensions
    :param Y: an array which has 1 dimension
    :param subset_size: subset_size (# of rows) of X and Y
    :param seed: random seed when taking a subset of X and Y 
    :return: the subset of X and Y
    """
    assert len(Y) >= subset_size, "Invalid subset_size of dataframe"
    assert len(X) == len(Y), "Inconsistent data size of input and target output"
    random_state = np.random.RandomState(seed)
    idcs = random_state.choice(len(Y), subset_size)
    return X[idcs, :], Y[idcs]


def append_dict_as_row(file_name: str, dict_of_elem: dict, field_names: Iterable) -> None:
    """
    A copied code from https://thispointer.com/python-how-to-append-a-new-row-to-an-existing-csv-file/
    Append "dict_of_elem" in csv file with "file_name". The columns of csv file are expected to be equal to
    "field_names".
    :param file_name: filename of csv file we want to append
    :param dict_of_elem: a dictionary that we want to append
    :param field_names: the keys of dictionary
    """
    # Open file in append mode
    with open(file_name, "a+", newline="") as write_obj:
        # Create a writer object from csv module
        dict_writer = DictWriter(write_obj, fieldnames=field_names)
        # Add dictionary in the csv
        dict_writer.writerow(dict_of_elem)


def fit(
        config: argparse.Namespace = None, use_wandb: bool = True, use_sweep: bool = True
) -> None:
    """
    Fit a random forest with "config.algorithm" to "config.dataset" with different hyper-parameters specified in
    "config" variable and log the result
    :param config: contains values needed to construct random forest and train it.
    :param use_wandb: whether to use wandb api or not
    :param use_sweep: whether to use the sweep method of wandb api or not
    """
    if use_wandb:
        wandb.init()
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        if use_sweep:
            config = wandb.config
        else:
            wandb.config = config

    # Set random seed
    np.random.seed(config.seed)
    random.seed(config.seed)

    # Setup dataset
    if config.dataset == "HEART":
        filepath = os.path.join("dataset", "new_heart_2020_cleaned.csv")
    else:
        raise NotImplementedError
    if not os.path.exists(filepath):
        preprocess_heart.main()

    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1].to_numpy()
    Y = df.iloc[:, -1].to_numpy()
    X, Y = get_subset(X, Y, config.sub_size, config.seed)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.20, random_state=config.seed
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.25, random_state=config.seed
    )

    # Setup model, train model, and find the val/test accuracy
    if config.algorithm == "SKLEARN":
        forest = RandomForestClassifier(
            n_estimators=config.n_estimators, max_depth=config.max_depth, bootstrap=config.bootstrap
        )
        forest.fit(X_train, Y_train)
        val_acc = np.sum(forest.predict(X_val) == Y_val) / len(Y_val)
        test_acc = np.sum(forest.predict(X_test) == Y_test) / len(Y_test)
    elif config.algorithm == "FASTFOREST":
        forest = Forest(
            X_train,
            Y_train,
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            bootstrap=config.bootstrap
        )
        forest.fit()
        val_acc = np.sum(forest.predict_batch(X_val)[0] == Y_val) / len(Y_val)
        test_acc = np.sum(forest.predict_batch(X_test)[0] == Y_test) / len(Y_test)
    else:
        raise NotImplementedError

    if config.verbose:
        print(f"{config.algorithm}'s random forest Validation Accuracy:", val_acc)
        print(f"{config.algorithm}'s random forest test Accuracy:", test_acc)

    # Log the results
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
            "verbose": config.verbose
        }
    )
    log_filename = os.path.join("hyperparams_sweep", "sweep_log.csv")
    if not os.path.exists(log_filename):  # Create a csv file if it doesn't exist
        os.makedirs("hyperparams_sweep", exist_ok=True)
        df = pd.DataFrame(columns=log_dict.keys())
        df.to_csv(log_filename, index=False)
    append_dict_as_row(log_filename, log_dict, log_dict.keys())


def main() -> None:
    """
    Compare the test accuracy of fastforest algorithm and sklearn algorithm with the choice of best hyper-parameters
    """
    log_filepath = os.path.join("hyperparams_sweep", "sweep_log.csv")
    filepath = os.path.join("dataset", "new_heart_2020_cleaned.csv")
    assert os.path.exists(filepath), "Heart disease dataset isn't available"
    assert os.path.exists(
        log_filepath
    ), "Can't find the log result of hyper-parameters sweeps"
    hyper_log = pd.read_csv(log_filepath)
    best_idx = np.argmax(hyper_log["val_acc"].to_numpy())
    config = hyper_log.iloc[best_idx]  # best hyper-parameter configuration

    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1].to_numpy()
    Y = df.iloc[:, -1].to_numpy()
    X, Y = get_subset(X, Y, 5000, config["seed"])
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.20, random_state=config["seed"]
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.25, random_state=config["seed"]
    )
    forest = Forest(
        X_train,
        Y_train,
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        bootstrap=config["bootstrap"]
    )
    forest.fit()
    val_acc = np.sum(forest.predict_batch(X_val)[0] == Y_val) / len(Y_val)
    test_acc = np.sum(forest.predict_batch(X_test)[0] == Y_test) / len(Y_test)

    algorithm = config["algorithm"]
    print(f"{algorithm}'s random forest Validation Accuracy:", config["val_acc"])
    print(f"{algorithm}'s random forest test Accuracy:", config["test_acc"])

    print("Fastforest's random forest Validation Accuracy:", val_acc)
    print("Fastforest's random forest test Accuracy:", test_acc)


if __name__ == '__main__':
    main()

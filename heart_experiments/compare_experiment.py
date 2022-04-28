import os
import numpy as np
import pandas as pd
import random
import scipy.stats as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, precision_score

from heart_experiments.fit_heart import split_data, load_data
from data_structures.forest import Forest


def find_best_hyper() -> pd.Series:
    """
    Find the best hyperparameters in hyperparams_sweep/sweep_log.csv in terms of val_acc of sklearn RF

    :return: return the best hyperparameters
    """
    log_filepath = os.path.join("hyperparams_sweep", "sweep_log.csv")
    assert os.path.exists(
        log_filepath
    ), "Can't find the log result of hyperparameters sweeps. Do hyperparameter sweeps!"
    hyper_log = pd.read_csv(log_filepath)
    best_idx = np.argmax(hyper_log["val_acc"].to_numpy())
    return hyper_log.iloc[best_idx]


def experiment_single() -> None:
    """
    Compare the test accuracy of fastforest and sklearn algorithm with best hyperparameters found in sweep_heart.py.
    """
    config = find_best_hyper()
    filepath = os.path.join("dataset", "new_heart_2020_cleaned.csv")
    assert os.path.exists(
        filepath
    ), "Heart disease dataset isn't available. Run preprocess_heart.py"
    X, Y = load_data(filepath)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(
        X, Y, [0.6, 0.2, 0.2], config["sub_size"], config["seed"], config["is_balanced"]
    )

    forest = Forest(
        X_train,
        Y_train,
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        bootstrap=config["bootstrap"],
    )

    forest.fit()
    val_acc = balanced_accuracy_score(Y_val, forest.predict_batch(X_val)[0])
    test_acc = balanced_accuracy_score(Y_test, forest.predict_batch(X_test)[0])

    algorithm = config["algorithm"]
    print(f"{algorithm}'s random forest Validation Accuracy:", config["val_acc"])
    print(f"{algorithm}'s random forest Test Accuracy:", config["test_acc"])

    print("Fastforest's random forest Validation Accuracy:", val_acc)
    print("Fastforest's random forest Test Accuracy:", test_acc)


def experiment_cf(
    t: int, seed: int, p_value: float = 0.95, verbose: bool = False
) -> bool:
    """
    Compare the test accuracy of fastforest and sklearn algorithm with best hyperparameters found in sweep_heart.py.
    Will compare their confidence interval by running each algorithm n times. It's a statistical hypothesis testing
    with p_value

    :param t: number of experiments
    :param seed: random seed used in fitting models
    :param p_value: p_value for hypothesis testing
    :param verbose: whether to print the ongoing fitting process
    :return: returns whether the confidence intervals of the two models are overlapped.
    """
    config = find_best_hyper()
    config["n_estimators"] = 10
    filepath = os.path.join("dataset", "new_heart_2020_cleaned.csv")
    assert os.path.exists(
        filepath
    ), "Heart disease dataset isn't available. Run preprocess_heart.py"
    X, Y = load_data(filepath)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(
        X, Y, [0.6, 0.2, 0.2], config["sub_size"], config["seed"], config["is_balanced"]
    )

    f_forest = Forest(
        X_train,
        Y_train,
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        bootstrap=config["bootstrap"],
    )
    sk_forest = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        bootstrap=config["bootstrap"],
    )

    ff_acc = np.empty(t)
    sk_acc = np.empty(t)

    np.random.seed(seed)
    random.seed(seed)
    for i in range(t):
        if verbose:
            print(f"--{i}th model is now training---")
        f_forest.fit()
        sk_forest.fit(X_train, Y_train)

        ff_acc[i] = balanced_accuracy_score(Y_test, f_forest.predict_batch(X_test)[0])
        sk_acc[i] = balanced_accuracy_score(Y_test, sk_forest.predict(X_test))
        ff_precision = precision_score(
            Y_test, f_forest.predict_batch(X_test)[0], average=None
        )
        sk_precision = precision_score(Y_test, sk_forest.predict(X_test), average=None)

        if verbose:
            print(f"The precision of fastforest is {ff_precision}")
            print(f"The precision of sklearn is {sk_precision}")

        # Reset models to fit and test for different bootstrap samples
        f_forest.trees = []
        f_forest.tree_feature_idcs = {}
        sk_forest = clone(sk_forest)  # return unfitted estimator

    if verbose:
        print(f"The balanced accuracy of fastforest is {ff_acc}")
        print(f"The balanced accuracy of sklearn is {sk_acc}")
    ff_mean = np.mean(ff_acc)
    sk_mean = np.mean(sk_acc)

    # ff_se = standard error of fastforest accuracy
    ff_se = np.std(ff_acc) / (t ** (1 / 2))
    # sk_se = standard error of sklearn accuracy
    sk_se = np.std(sk_acc) / (t ** (1 / 2))

    print(
        f"The mean balanced accuracy of fastforest is {ff_mean}, and its standard error is {ff_se}."
    )
    print(
        f"The mean balanced accuracy of sklearn is {sk_mean}, and its standard error is {sk_se}."
    )

    confidence_level = 1 - p_value
    if t >= 30:
        z_score = st.norm.ppf(1 - 1 / 2 * p_value)
    else:
        z_score = st.t.ppf(confidence_level, t - 1)
    ff_interval = (ff_mean - z_score * ff_se, ff_mean + z_score * ff_se)
    sk_interval = (sk_mean - z_score * sk_se, sk_mean + z_score * sk_se)

    # Check whether their confidence interval is overlapped
    overlap = 2 * (z_score * ff_se + z_score * sk_se) < (ff_mean - sk_mean)
    print("-" * 30)
    print(
        f"The confidence interval of fast forest is {ff_interval} at {confidence_level * 100}% confidence level."
    )
    print(
        f"The confidence interval of sklearn is {sk_interval} at {confidence_level * 100}% confidence level."
    )

    print(f"It's {overlap} that the two confidence_intervals are overlapped.")
    return overlap


if __name__ == "__main__":
    experiment_cf(30, 10, 0.05, True)

import os
import numpy as np
import pandas as pd
import random
import scipy.stats as st

from fit_heart import get_subset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from forest import Forest


def find_best_hyper() -> pd.Series:
    """
    Find the best hyper-parameters in hyperparams_sweep/sweep_log.csv in terms of val_acc of sklearn RF
    :return: return the best hyper-parameters
    """
    log_filepath = os.path.join("hyperparams_sweep", "sweep_log.csv")
    assert os.path.exists(
        log_filepath
    ), "Can't find the log result of hyper-parameters sweeps. Do hyper-parameter sweeps!"
    hyper_log = pd.read_csv(log_filepath)
    best_idx = np.argmax(hyper_log["val_acc"].to_numpy())
    config = hyper_log.iloc[best_idx]  # best hyper-parameter configuration

    return config


def experiment_single() -> None:
    """
    Compare the test accuracy of fastforest and sklearn algorithm with best hyper-parameters found in sweep_heart.py.
    Will compare it by running each algorithm once
    """
    # Find the best hyper-parameters
    config = find_best_hyper()

    # Setup dataset
    filepath = os.path.join("dataset", "new_heart_2020_cleaned.csv")
    assert os.path.exists(
        filepath
    ), "Heart disease dataset isn't available. Run preprocess_heart.py"
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

    # Setup model
    forest = Forest(
        X_train,
        Y_train,
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        bootstrap=config["bootstrap"],
    )

    # Train and print the result
    forest.fit()
    val_acc = np.sum(forest.predict_batch(X_val)[0] == Y_val) / len(Y_val)
    test_acc = np.sum(forest.predict_batch(X_test)[0] == Y_test) / len(Y_test)

    algorithm = config["algorithm"]
    print(f"{algorithm}'s random forest Validation Accuracy:", config["val_acc"])
    print(f"{algorithm}'s random forest test Accuracy:", config["test_acc"])

    print("Fastforest's random forest Validation Accuracy:", val_acc)
    print("Fastforest's random forest test Accuracy:", test_acc)


def experiment_cf(n: int, seed: int, p_value: float = 0.95, verbose: bool = False) -> bool:
    """
    Compare the test accuracy of fastforest and sklearn algorithm with best hyper-parameters found in sweep_heart.py.
    Will compare their confidence interval by running each algorithm n times (statistical hypothesis testing
    with p_value)
    :param n: number of experiments
    :param seed: random seed used in fitting models
    :param p_value: p_value for hypothesis testing
    :param verbose: whether to print the ongoing fitting process
    :return: returns True if the confidence intervals of two models are overlapped. Returns False otherwise.
    """
    # Setup hyper-parameters
    config = find_best_hyper()

    # Setup dataset
    filepath = os.path.join("dataset", "new_heart_2020_cleaned.csv")
    assert os.path.exists(
        filepath
    ), "Heart disease dataset isn't available. Run preprocess_heart.py"
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

    # Setup model
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

    # Train and find the confidence interval
    ff_acc = np.zeros(n)
    sk_acc = np.zeros(n)

    np.random.seed(seed)
    random.seed(seed)
    for i in range(n):
        print(f"--{i}th model is now training---")
        f_forest.fit()
        sk_forest.fit(X_train, Y_train)

        # Calculate and store the accuracy of models
        ff_acc[i] = np.sum(f_forest.predict_batch(X_test)[0] == Y_test) / len(Y_test)
        sk_acc[i] = np.sum(sk_forest.predict(X_test) == Y_test) / len(Y_test)

        # Reset models
        f_forest.trees = []
        sk_forest = clone(sk_forest)  # return unfitted estimator

    ff_mean = np.mean(ff_acc)
    sk_mean = np.mean(sk_acc)
    ff_se = np.std(ff_acc) / (n ** (1 / 2))  # ff_se = standard error of fastforest accuracy
    sk_se = np.std(sk_acc) / (n ** (1 / 2))  # sk_se = standard error of sklearn accuracy

    print(f"The mean accuracy of fastforest is {ff_mean}, and its standard error is {ff_se}.")
    print(f"The mean accuracy of sklearn is {sk_mean}, and its standard error is {sk_se}.")
    print(ff_acc)
    print(sk_acc)
    if n>=30:
        z_score = st.norm.ppf(p_value)
    else:
        z_score = st.t.ppf(p_value, n-1)
    ff_interval = (ff_mean - z_score * ff_se, ff_mean + z_score * ff_se)
    sk_interval = (sk_mean - z_score * sk_se, sk_mean + z_score * sk_se)

    is_same = 2 * (z_score * ff_se + z_score * sk_se) < (ff_mean - sk_mean) # Check whether their
    # confidence interval is overlapped
    print("-" * 30)
    print(f"The confidence interval of fast forest is {ff_interval} at {p_value * 100}% confidence level.")
    print(f"The confidence interval of sklearn is {sk_interval} at {p_value * 100}% confidence level.")

    print(f"It's {is_same} that the two confidence_intervals are overlapped.")
    return is_same


if __name__ == "__main__":
    experiment_cf(30, 10, 0.99)

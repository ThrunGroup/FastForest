import pandas as pd
import numpy as np
import time
from typing import Any, Union

from data_structures.wrappers.histogram_random_forest_classifier import HistogramRandomForestClassifier as HRFC
from data_structures.wrappers.histogram_random_forest_regressor import HistogramRandomForestRegressor as HRFR
from utils.constants import MAB, EXACT, FLIGHT, AIR, APS, BLOG
from experiments.datasets.data_loader import fetch_data


def experiment(is_classification: bool, model: callable, dataset: str):
    """
    Run a single experiment compute runtimes and number of histogram insertions for a single model, datasets, and seed.
    Marries some of the code in compare_runtimes.py and compare_budgets.py for quicker analysis

    :param is_classification: boolean, self-explanatory
    :param model: model to train
    :param dataset: datasets to use
    :return: None, prints output
    """
    print(f"\nRunning {dataset} dataset")
    X_train, y_train, X_test, y_test = fetch_data(dataset)
    mab_model = model(
        max_leaf_nodes=10,
        max_depth=100,
        solver=MAB,
        n_estimators=5,
        verbose=False,
    )
    exact_model = model(
        max_leaf_nodes=10,
        max_depth=100,
        solver=EXACT,
        n_estimators=5,
        verbose=False,
    )

    for str_ in ["MAB", "EXACT"]:
        if str_ == "MAB":
            model = mab_model
        elif str_ == "EXACT":
            model = exact_model

        print("\nFitting " + str_ + " model")
        start = time.time()
        model.fit(data=X_train, labels=y_train)
        print("time taken: ", time.time() - start)
        if is_classification:
            print("Accuracy: ", np.mean(model.predict_batch(X_test)[0] == y_test))
        else:
            print(f"MSE: {np.mean(np.square((model.predict_batch(X_test) - y_test)))}")
        print("Number of queries:", model.num_queries)


if __name__ == "__main__":
    # Classification
    datasets = [APS, FLIGHT]
    for dataset in datasets:
        experiment(True, HRFC, dataset)

    # Regression
    datasets = [AIR, BLOG]
    for dataset in datasets:
        experiment(False, HRFR, dataset)

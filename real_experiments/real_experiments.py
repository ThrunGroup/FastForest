import pandas as pd
import numpy as np
import time
from typing import Any, Union

from data_structures.wrappers.histogram_random_forest_classifier import HistogramRandomForestClassifier as HRFC
from data_structures.wrappers.histogram_random_forest_regressor import HistogramRandomForestRegressor as HRFR
from utils.constants import MAB, EXACT, FLIGHT, AIR, APS, BLOG
from dataset import fetch_data


def experiment(is_classification: bool, model: callable, dataset: str):
    print(f"\nSTART EXPERIMENT on ***{dataset}***")
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
    print("\nStart fitting mab model")
    start = time.time()
    mab_model.fit(data=X_train, labels=y_train)
    print("time taken: ", time.time() - start)
    if is_classification:
        print("acc: ", np.mean(mab_model.predict_batch(X_test)[0] == y_test))
    else:
        print(f"mse: {np.mean(np.square((mab_model.predict_batch(X_test) - y_test)))}")
    print("number of queries:", mab_model.num_queries)

    print("\nStart fitting exact model")
    start = time.time()
    exact_model.fit(X_train, y_train)
    print("time taken: ", time.time() - start)
    if is_classification:
        print("acc: ", np.mean(exact_model.predict_batch(X_test)[0] == y_test))
    else:
        print(f"mse: {np.mean(np.square((exact_model.predict_batch(X_test) - y_test)))}")
    print("number of queries:", exact_model.num_queries)


if __name__ == "__main__":
    # Classification
    datasets = [FLIGHT, APS]
    for dataset in datasets:
        experiment(True, HRFC, dataset)

    # Regression
    datasets = [AIR, BLOG]
    for dataset in datasets:
        experiment(False, HRFR, dataset)

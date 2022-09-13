from sklearn.datasets import fetch_california_housing
import time
import numpy as np

from data_structures.wrappers.histogram_random_forest_regressor import HistogramRandomForestRegressor as HRFR
from utils.constants import MAB, EXACT

from experiments.datasets.data_loader import get_air_data


if __name__ == "__main__":
    data, target, test_data, test_target = get_air_data()
    mab_model = HRFR(
        data=data,
        labels=target,
        max_leaf_nodes=5,
        max_depth=100,
        n_estimators=5,
        solver=MAB,
        batch_size=1000,
        verbose=True,
    )
    exact_model = HRFR(
        data=data,
        labels=target,
        max_leaf_nodes=5,
        max_depth=100,
        n_estimators=5,
        solver=EXACT,
        verbose=True,
    )

    print("Data shape: ", data.shape)
    for str_ in ["MAB", "EXACT"]:
        if str_ == "MAB":
            model = mab_model
        elif str_ == "EXACT":
            model = exact_model

        print("Fitting " + str_ + " model")
        start = time.time()
        model.fit()
        print(f"Fitting time: {time.time() - start}")
        model.predict_batch(data) - target
        print(f"MSE: {np.mean(np.square((model.predict_batch(test_data) - test_target)))}")
        print(f"Number of queries: {model.num_queries}")

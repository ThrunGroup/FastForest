from sklearn.datasets import fetch_california_housing
import time
import numpy as np

from data_structures.wrappers.histogram_random_forest_regressor import HistogramRandomForestRegressor as HRFR
from utils.constants import MAB, EXACT

from experiments.datasets.data_loader import get_air_data


if __name__ == "__main__":
    # housing_data = fetch_california_housing()
    # data = housing_data.data
    # target = housing_data.target
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
    print(data.shape)
    start = time.time()
    print("fitting MAB model")
    mab_model.fit()
    print(f"fitting time: {time.time() - start}")
    mab_model.predict_batch(data) - target
    print(f"Mse: {np.mean(np.square((mab_model.predict_batch(test_data)-test_target)))}")
    print(f"Num_queries: {mab_model.num_queries}")

    start = time.time()
    print("fitting EXACT model")
    exact_model.fit()
    print(f"fitting time: {time.time() - start}")
    exact_model.predict_batch(data) - target
    print(f"Mse: {np.mean(np.square((exact_model.predict_batch(test_data)-test_target)))}")
    print(f"Num_queries: {exact_model.num_queries}")
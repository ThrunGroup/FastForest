from sklearn import datasets
import numpy as np

from data_structures.tree_classifier import TreeClassifier
from data_structures.wrappers.random_forest_classifier import RandomForestClassifier
import utils.utils

import matplotlib.pyplot as plt
import numpy as np
import random
from utils.utils import class_to_idx
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes

from data_structures.tree_regressor import TreeRegressor
from data_structures.forest_regressor import ForestRegressor
from utils.constants import SQRT, EXACT, MAB, LINEAR


def test_forest_diabetes(
    seed: int = 1,
    verbose: bool = False,
    features_subsampling: str = None,
    solver: str = MAB,
    with_replacement: bool = False,
    print_sklearn: bool = False,
    boosting: bool = False,
):
    if verbose:
        print("--RF experiment with diabetes datasets--")
    diabetes = load_diabetes()
    data, labels = diabetes.data, diabetes.target
    if print_sklearn:
        max_features = "sqrt" if features_subsampling == SQRT else features_subsampling
        RF = RandomForestRegressor(
            n_estimators=10, max_depth=6, max_features=max_features, random_state=seed
        )
        RF.fit(data, labels)
        print("-sklearn forest")
        mse = np.sum(np.square(RF.predict(data) - labels)) / len(data)
        print(f"MSE of sklearn forest is {mse}\n")

    FF = ForestRegressor(
        n_estimators=10,
        max_depth=6,
        verbose=verbose,
        feature_subsampling=features_subsampling,
        random_state=seed,
        with_replacement=with_replacement,
        bin_type="",
        solver=solver,
        boosting=boosting,
    )
    FF.fit(data, labels)
    mse = np.sum(np.square(FF.predict_batch(data) - labels)) / len(data)
    if verbose:
        print("-FastForest")
        print(f"Features subsampling: {features_subsampling}")
        print(f"Seed : {seed}")
        print(f"Solver: {solver}")
        print(f"Sample with replacement: {with_replacement}")
        print(f"MSE of fastforest is {mse}")
        print(f"num_queries is {FF.num_queries}\n")
    return FF.num_queries, mse


if __name__ == "__main__":
    test_forest_diabetes()

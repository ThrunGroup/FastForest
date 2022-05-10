import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes

from data_structures.tree_regressor import TreeRegressor
from data_structures.forest_regressor import ForestRegressor
from utils.constants import SQRT, EXACT, MAB


def test_tree_diabetes(
    seed: int = 1, verbose: bool = False, with_replacement: bool = False, solver: str = MAB, print_sklearn: bool = False
):
    print("--DT experiment with diabetes dataset--")
    np.random.seed(seed)
    random.seed(seed)
    diabetes = load_diabetes()
    data, labels = diabetes.data, diabetes.target
    if print_sklearn:
        DT = DecisionTreeRegressor(max_depth=6, random_state=seed)
        DT.fit(data, labels)
        print("-Sklearn")
        if verbose:
            print(export_text(DT))
            plot_tree(DT)
            plt.show()
        mse = np.sum(np.square(DT.predict(data) - labels)) / len(data)
        print(f"MSE is {mse}\n")

    print("-FastTree")
    print(f"Seed : {seed}")
    print(f"Solver: {solver}")
    print(f"Sample with replacement: {with_replacement}")
    tree = TreeRegressor(
        data,
        labels,
        max_depth=6,
        verbose=verbose,
        random_state=seed,
        solver=solver,
        with_replacement=with_replacement,
    )
    tree.fit()
    if verbose:
        tree.tree_print()
    mse = np.sum(np.square(tree.predict_batch(data) - labels)) / len(data)
    print(f"MSE is {mse}")
    print(f"num_queries is {tree.num_queries}\n")


def test_forest_diabetes(
    seed: int = 1,
    verbose: bool = False,
    features_subsampling: str = None,
    solver: str = MAB,
    with_replacement: bool = False,
    print_sklearn: bool = False
):
    print("--RF experiment with diabetes dataset--")
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

    print("-FastForest")
    print(f"Features subsampling: {features_subsampling}")
    print(f"Seed : {seed}")
    print(f"Solver: {solver}")
    print(f"Sample with replacement: {with_replacement}")
    FF = ForestRegressor(
        n_estimators=10,
        max_depth=6,
        verbose=verbose,
        feature_subsampling=features_subsampling,
        random_state=seed,
        with_replacement=with_replacement,
        solver=solver,
    )
    FF.fit(data, labels)
    mse = np.sum(np.square(FF.predict_batch(data) - labels)) / len(data)
    print(f"MSE of fastforest is {mse}")
    print(f"num_queries is {FF.num_queries}\n")


if __name__ == "__main__":
    test_tree_diabetes(with_replacement=True)
    test_tree_diabetes(with_replacement=False)
    test_forest_diabetes(seed=50, with_replacement=True)
    test_forest_diabetes(seed=50, with_replacement=False)
    test_forest_diabetes(seed=50, features_subsampling=SQRT, with_replacement=True)
    test_forest_diabetes(seed=50, features_subsampling=SQRT, with_replacement=False)

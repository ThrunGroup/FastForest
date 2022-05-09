import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes

from data_structures.tree_regressor import TreeRegressor
from data_structures.forest_regressor import ForestRegressor
from utils.constants import SQRT


def test_tree_diabetes(seed: int = 1, verbose: bool = False):
    np.random.seed(seed)
    random.seed(seed)
    diabetes = load_diabetes()
    data, labels = diabetes.data, diabetes.target

    DT = DecisionTreeRegressor(max_depth=6, random_state=seed)
    DT.fit(data, labels)
    print("---Ground_Truth_Tree---")
    if verbose:
        print(export_text(DT))
        plot_tree(DT)
        plt.show()
    mse = np.sum(np.square(DT.predict(data) - labels)) / len(data)
    print(f"MSE is {mse}\n")

    print("---FastTree---")
    tree = TreeRegressor(data, labels, max_depth=6, verbose=verbose, random_state=seed)
    tree.fit()
    if verbose:
        tree.tree_print()
    mse = np.sum(np.square(tree.predict_batch(data) - labels)) / len(data)
    print(f"MSE is {mse}\n")


def test_forest_diabetes(
    seed: int = 1, verbose: bool = False, features_subsampling: str = None
):
    diabetes = load_diabetes()
    data, labels = diabetes.data, diabetes.target

    print(f"Experiment with {features_subsampling} feature_subsampling")
    max_features = "sqrt" if features_subsampling == SQRT else features_subsampling
    RF = RandomForestRegressor(
        n_estimators=10, max_depth=6, max_features=max_features, random_state=seed
    )
    RF.fit(data, labels)
    print("---Ground_Truth_Forest---")
    mse = np.sum(np.square(RF.predict(data) - labels)) / len(data)
    print(f"MSE of sklearn forest is {mse}\n")

    print("---FastForest---")
    FF = ForestRegressor(
        n_estimators=10,
        max_depth=6,
        verbose=verbose,
        feature_subsampling=features_subsampling,
        random_state=seed,
    )
    FF.fit(data, labels)
    mse = np.sum(np.square(FF.predict_batch(data) - labels)) / len(data)
    print(f"MSE of fastforest is {mse}\n")


if __name__ == "__main__":
    test_tree_diabetes()
    test_forest_diabetes()
    test_forest_diabetes(features_subsampling=SQRT)

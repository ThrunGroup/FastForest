import matplotlib.pyplot as plt
import numpy as np

from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes

import ground_truth
from data_structures.tree_regressor import TreeRegressor
from data_structures.forest_regressor import ForestRegressor


def test_tree_diabetes(verbose: bool = False):
    diabetes = load_diabetes()
    data, labels = diabetes.data, diabetes.target

    DT = DecisionTreeRegressor(max_depth=6)
    DT.fit(data, labels)
    print("---Ground_Truth_Tree---")
    if verbose:
        print(export_text(DT))
        plot_tree(DT)
        plt.show()
    mse = np.sum(np.square(DT.predict(data) - labels)) / len(data)
    print(f"MSE is {mse}")

    print("---FastTree---")
    tree = TreeRegressor(data, labels, max_depth=6, verbose=verbose)
    tree.fit()
    if verbose:
        tree.tree_print()
    mse = np.sum(np.square(tree.predict_batch(data) - labels)) / len(data)
    print(f"MSE is {mse}")


def test_forest_diabetes(verbose: bool = False):
    diabetes = load_diabetes()
    data, labels = diabetes.data, diabetes.target

    RF = RandomForestRegressor(n_estimators=1, max_depth=6)
    RF.fit(data, labels)
    print("---Ground_Truth_Tree---")
    mse = np.sum(np.square(RF.predict(data) - labels)) / len(data)
    print(f"MSE of sklearn forest is {mse}")

    print("---FastTree---")
    FF = ForestRegressor(n_estimators=20, max_depth=1, verbose=verbose)
    FF.fit(data, labels)
    mse = np.sum(np.square(FF.predict_batch(data) - labels)) / len(data)
    print(f"MSE of fastforest is {mse}")


if __name__ == "__main__":
    test_tree_diabetes()
    test_forest_diabetes()

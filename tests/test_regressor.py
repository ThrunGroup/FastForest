from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import numpy as np

from data_structures.tree_regressor import TreeRegressor


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
    tree = TreeRegressor(data, labels, max_depth=6)
    tree.fit()
    if verbose:
        tree.tree_print()
    mse = np.sum(np.square(tree.predict_batch(data) - labels)) / len(data)
    print(f"MSE is {mse}")


if __name__ == "__main__":
    test_tree_diabetes()

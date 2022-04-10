import numpy as np
import sys
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

from data_generator import create_data
from histogram import Histogram
from fast_forest import get_impurity_reductions, solve_mab
from tree import Tree


def ground_truth_stump(X: np.ndarray, show: bool = False):
    """
    Given a dataset, perform the first step of making a tree: find the single best (feature, feature_value) pair
    to split on using the sklearn implementation.

    :param X: Dataset to build a stump out of
    :param show: Whether to display the visual plot
    :return: None
    """
    # Since y is mostly correlated with the first feature, we expect a 1-node stump to look at the first feature
    # and choose that. So if x0 < 0.5, choose 0, otherwise choose 1
    DT = DecisionTreeClassifier(max_depth=1)
    DT.fit(X[:, :2], X[:, 2])
    print(export_text(DT))
    if show:
        plot_tree(DT)
        plt.show()


def main():
    X = create_data(10000)
    ground_truth_stump(X, show=False)
    h = Histogram(0, num_bins=11)
    h.add(X, X[:, -1])

    reductions, vars = get_impurity_reductions(h, np.arange(len(h.bin_edges)), ret_vars=True)
    print("=> THIS IS GROUND TRUTH\n")
    print(reductions)
    print(vars)
    print(np.argmin(reductions))
    # print(h[0])
    print("\n\n")

    print("=> THIS IS MAB\n")
    data = X[:, :2]
    labels = X[:, 2]
    print("best arm is: ", solve_mab(data, labels))
    t = Tree(X[:, 0:2], X[:, 2], max_depth=3)
    t.fit()
    t.tree_print()

if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    np.random.seed(0)
    main()

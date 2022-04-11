import numpy as np
import sys
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

from data_generator import create_data
from histogram import Histogram
from fast_forest import get_impurity_reductions, solve_mab
from tree import Tree

import sklearn.datasets


def ground_truth_tree(
    data: np.ndarray, labels: np.ndarray, max_depth: int = 1, show: bool = False
):
    """
    Given a dataset, perform the first step of making a tree: find the single best (feature, feature_value) pair
    to split on using the sklearn implementation.

    :param X: Dataset to build a stump out of
    :param show: Whether to display the visual plot
    :return: None
    """
    DT = DecisionTreeClassifier(max_depth=max_depth)
    DT.fit(data, labels)
    print(export_text(DT))
    if show:
        plot_tree(DT)
        plt.show()


def test_iris_agreement() -> None:
    iris = sklearn.datasets.load_iris()

    two_class_idcs = np.where((iris.target == 2) | (iris.target == 1))
    two_class_data = iris.data[two_class_idcs]
    two_class_labels = iris.target[two_class_idcs]
    two_class_labels[np.where(two_class_labels == 2)] = 0

    # Note: currently only support 2-class target
    ground_truth_tree(
        data=two_class_data, labels=two_class_labels, max_depth=5, show=True
    )
    t = Tree(data=two_class_data, labels=two_class_labels, max_depth=5)
    t.fit()
    t.tree_print()


def main():
    X = create_data(10000)
    data = X[:, :-1]
    labels = X[:, -1]

    ground_truth_tree(data, labels, show=False)
    h = Histogram(0, num_bins=11)
    h.add(data, labels)

    reductions, vars = get_impurity_reductions(
        h, np.arange(len(h.bin_edges)), ret_vars=True
    )
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
    t = Tree(data, labels, max_depth=3)
    t.fit()
    t.tree_print()

    print("\n\nTesting iris dataset agreement:")
    test_iris_agreement()


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    np.random.seed(0)
    main()

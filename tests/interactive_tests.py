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


def test_tree_iris2(verbose: bool = False) -> None:
    """
    Compare tree fitting with different hyperparameters
    :param verbose: Whether to print how the trees are constructed in details.
    """
    iris = datasets.load_iris()
    data, labels = iris.data, iris.target
    classes_arr = np.unique(labels)
    classes = class_to_idx(classes_arr)

    def test_tree(max_leaf_nodes, bin_type, print_str):
        print("-" * 30)
        print(f"Fitting {print_str} tree")
        t = TreeClassifier(
            data=data,
            labels=labels,
            max_depth=5,
            classes=classes,
            budget=None,
            max_leaf_nodes=max_leaf_nodes,
            bin_type=bin_type,
        )
        t.fit()
        if verbose:
            t.tree_print()
        acc = np.sum(t.predict_batch(data)[0] == labels)
        print("MAB solution Tree Train Accuracy:", acc / len(data))

    # Depth-first tree
    test_tree(None, "", "Depth-first splitting")

    # Best-first tree
    test_tree(32, "", "Best-first splitting")

    # Linear bin tree
    test_tree(None, LINEAR, "Linear bin splitting")


if __name__ == "__main__":
    test_tree_iris2()

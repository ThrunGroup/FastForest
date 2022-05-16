import sklearn.datasets
import numpy as np

from data_structures.tree_classifier import TreeClassifier
from data_structures.wrappers.random_forest_classifier import RandomForestClassifier
import utils.utils

import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes

from data_structures.tree_regressor import TreeRegressor
from data_structures.forest_regressor import ForestRegressor
from utils.constants import SQRT, EXACT, MAB


def test_zero_budget_tree_iris() -> None:
    iris = sklearn.datasets.load_iris()
    data, labels = iris.data, iris.target
    classes_arr = np.unique(labels)
    classes = utils.utils.class_to_idx(classes_arr)
    t = TreeClassifier(data=data, labels=labels, max_depth=5, classes=classes, budget=0)
    t.fit()
    t.tree_print()
    acc = np.sum(t.predict_batch(data)[0] == labels)
    print("MAB solution Tree Train Accuracy:", acc / len(data))


def test_increasing_budget_tree_iris() -> None:
    iris = sklearn.datasets.load_iris()
    data, labels = iris.data, iris.target
    classes_arr = np.unique(labels)
    classes = utils.utils.class_to_idx(classes_arr)

    t1 = TreeClassifier(
        data=data, labels=labels, max_depth=5, classes=classes, budget=50
    )
    t1.fit()
    t1.tree_print()
    print("T1 Number of queries:", t1.num_queries)
    acc1 = np.sum(t1.predict_batch(data)[0] == labels)
    print()
    print()
    t2 = TreeClassifier(
        data=data, labels=labels, max_depth=5, classes=classes, budget=1000
    )
    t2.fit()
    t2.tree_print()

    print("T2 Number of queries:", t2.num_queries)
    acc2 = np.sum(t2.predict_batch(data)[0] == labels)
    print(acc1, acc2)


def test_wrapper_forest_iris() -> None:
    iris = sklearn.datasets.load_iris()
    data, labels = iris.data, iris.target
    f = RandomForestClassifier(
        data=data,
        labels=labels,
        n_estimators=20,
        max_depth=5,
    )
    f.fit()
    acc = np.sum(f.predict_batch(data)[0] == labels)
    print("Accuracy of wrapper:", (acc / len(data)))


def test_tree_diabetes(
    seed: int = 1,
    verbose: bool = False,
    with_replacement: bool = False,
    solver: str = MAB,
    print_sklearn: bool = False,
):
    if verbose:
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

    tree = TreeRegressor(
        data,
        labels,
        max_depth=6,
        verbose=verbose,
        random_state=seed,
        solver=solver,
        bin_type="",
        with_replacement=with_replacement,
    )
    tree.fit()
    if verbose:
        tree.tree_print()
    mse = np.sum(np.square(tree.predict_batch(data) - labels)) / len(data)
    if verbose:
        print("-FastTree")
        print(f"Seed : {seed}")
        print(f"Solver: {solver}")
        print(f"Sample with replacement: {with_replacement}")
        print(f"MSE is {mse}")
        print(f"num_queries is {tree.num_queries}\n")
    return tree.num_queries, mse


if __name__ == "__main__":
    # test_zero_budget_tree_iris()
    # test_increasing_budget_tree_iris()
    # test_wrapper_forest_iris()
    test_tree_diabetes()

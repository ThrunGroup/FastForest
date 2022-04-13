import numpy as np
import sys
import matplotlib.pyplot as plt
import sklearn.datasets

from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree,
    export_text,
)

from data_generator import create_data
from mab_functions import solve_mab
from tree import Tree
from forest import Forest


def ground_truth_tree(
    data: np.ndarray, labels: np.ndarray, max_depth: int = 1, show: bool = False
) -> None:
    """
    Given a dataset, create the ground truth tree using sklearn.
    If max_depth = 1, perform the first step of making a tree: find the single best (feature, feature_value) pair
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

    acc = np.sum(DT.predict(data) == labels) / len(data)
    print("Ground truth tree Train Accuracy:", acc)


def ground_truth_forest(
    data: np.ndarray,
    labels: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 5,
    n_classes: int = 2,
) -> None:
    """
    Given a dataset, create the ground truth tree using sklearn.
    If n_estimators = 1, fits only the first ree
    :param data: data to fit
    :param labels: labels of the data
    :param max_depth: max depth of an individual tree
    :param show: whether to show the random forest using matplotlib
    :return: None
    """
    RF = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,)
    RF.fit(data, labels)
    acc = np.sum(RF.predict(data) == labels) / len(data)
    print("Ground truth random forest Train Accuracy:", acc)


def reduce_to_2class(
    data: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    two_class_idcs = np.where((labels == 2) | (labels == 1))
    two_class_data = data[two_class_idcs]
    two_class_labels = labels[two_class_idcs]
    two_class_labels[np.where(two_class_labels == 2)] = 0
    return two_class_data, two_class_labels


def test_tree_iris() -> None:
    iris = sklearn.datasets.load_iris()
    data, labels = iris.data, iris.target
    num_classes = len(set(labels))

    # Note: currently only support 2-class target
    ground_truth_tree(data=data, labels=labels, max_depth=5, show=False)
    t = Tree(data=data, labels=labels, max_depth=5)
    t.fit()
    t.tree_print()
    acc = np.sum(t.predict_batch(data)[0] == labels)
    print("MAB solution Tree Train Accuracy:", acc / len(data))


def test_forest_iris() -> None:
    iris = sklearn.datasets.load_iris()
    data, labels = iris.data, iris.target
    num_classes = len(set(labels))

    ground_truth_forest(
        data=data, labels=labels, n_estimators=100, max_depth=5, n_classes=num_classes
    )

    f = Forest(
        data=data, labels=labels, n_estimators=20, max_depth=5, n_classes=num_classes
    )
    f.fit()
    acc = np.sum(f.predict_batch(data)[0] == labels)
    print("MAB solution Forest Train Accuracy:", acc / len(data))


def test_forest_digits() -> None:
    digits = sklearn.datasets.load_digits()
    data, labels = digits.data, digits.target
    num_classes = len(set(labels))

    ground_truth_forest(
        data=data, labels=labels, n_estimators=5, max_depth=5, n_classes=num_classes
    )

    f = Forest(
        data=data, labels=labels, n_estimators=5, max_depth=5, n_classes=num_classes
    )
    f.fit()
    acc = np.sum(f.predict_batch(data)[0] == labels)
    print("MAB solution Forest Train Accuracy:", acc / len(data))


def test_tree_toy(show: bool = False) -> None:
    X = create_data(10000)
    data = X[:, :-1]
    labels = X[:, -1]

    print("=> Ground truth:\n")
    ground_truth_tree(data, labels, show=show)

    print("\n\n=> MAB:\n")
    print("Best arm from solve_mab is: ", solve_mab(data, labels))

    print("\n\n=> Tree fitting:")
    t = Tree(data, labels, max_depth=3)
    t.fit()
    t.tree_print()


def main():
    print("Testing toy data decision stump:")
    test_tree_toy(show=False)

    print("\n" * 4)
    print("Testing tree iris dataset agreement:")
    test_tree_iris()

    print("\n" * 4)
    print("Testing forest iris dataset agreement:")
    test_forest_iris()

    print("\n" * 4)
    print("Testing forest digit dataset agreement:")
    test_forest_digits()


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    np.random.seed(0)
    main()

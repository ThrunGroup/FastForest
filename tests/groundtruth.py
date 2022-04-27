import numpy as np
import matplotlib.pyplot as plt


from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree,
    export_text,
)


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
    RF = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
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
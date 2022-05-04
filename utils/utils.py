import numpy as np
from collections import defaultdict
from typing import DefaultDict


def type_check() -> None:
    """
    Helper function for type checking.
    We need to do this below to avoid the circular import: Tree <--> Node
    See https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/

    :return: None
    """
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from tree import Tree


def count_occurrence(class_: np.ndarray, labels: np.ndarray) -> int:
    """
    Helper function for counting the occurrence of class_ in labels

    :param class_: class name to count
    :param labels: labels of the dataset
    :return: number of datapoints with the given class name
    """
    return len(np.where(labels == class_)[0])


def class_to_idx(classes: np.ndarray,) -> dict:
    """
    Helpful function for generating dictionary that maps class names to class index
    Helper function for function for generating dictionary that maps class names to class index

    :param classes: A list of unique class names for the dataset
    :return: A dictionary from class names to class indices
    """
    return dict(zip(classes, range(len(classes))))


def counts_of_labels(class_dict: dict, labels: np.ndarray) -> np.ndarray:
    """
    Helper function for generating counts array.


    :param: class_dict: dict from class name to class index
    :param labels: labels of dataset
    :return: array of counts of each class label, indexed by class index
    """
    classes = np.unique(labels)
    counts = np.zeros(len(class_dict))
    for class_ in classes:
        class_idx = class_dict[class_]
        counts[class_idx] = count_occurrence(class_, labels)
    return counts


def data_to_discrete(data: np.ndarray, n: int) -> DefaultDict:
    """
    Helpful function for creating a dictionary of unique values of discrete features
    Ex) data = np.array([[0.1 1],
                         [0.2 2],
                         [0.3 3],
                         [0.4 2],
                         [0.5 1],
                         [0.6 2],
                         [0.7 1],
                         [0.8 3]])
        data_to_discrete(data, n=3) returns a dictionary {1: np.array([1,2,3])}. 1 is a feature index and
        [1,2,3] is discrete feature values of feature index 1.

    :param data: An input data array with 2 dimension
    :param n: An integer number which is a criteria for deciding whether some features are discrete or not
    :return: A dictionary mapping from discrete feature index to the list of its unique feature
    values
    """
    discrete_dict = defaultdict(list)
    for feature_idx in range(len(data[0])):
        unique_fvals = np.unique(data[:, feature_idx])
        if len(unique_fvals) <= n: # If not, "feature_idx"th feature is not discrete
            discrete_dict[feature_idx] = unique_fvals
    return discrete_dict

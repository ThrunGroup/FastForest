import numpy as np


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


def class_to_idx(
    classes: np.ndarray,
) -> dict:
    """
    Helper function for function for generating dictionary that maps class names to class index

    :param classes: A list of unique class names for the dataset
    :return: A dictionary from class names to class indices
    """
    return dict(zip(classes, range(len(classes))))


def counts_of_labels(class_dict: dict, labels: np.ndarray) -> np.ndarray:
    """
    Helper function for generating counts array.
    Counts is a numpy array that stores counts of the classes in labels.

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

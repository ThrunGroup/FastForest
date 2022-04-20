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
    Helpful for function for counting the occurrence of class_ in labels
    """
    return len(np.where(labels == class_)[0])


def class_to_idx(classes: np.ndarray,) -> dict:
    """
    Helpful for function for generating dictionary that maps class names to class index
    """
    return dict(zip(classes, range(len(classes))))


def counts_on_labels(class_dict: dict, labels: np.ndarray) -> np.ndarray:
    """
    Helper function for generating counts array.
    Counts is a numpy array that stores counts of the classes in labels.
    :return: counts
    """
    classes = np.unique(labels)
    counts = np.zeros(len(class_dict))
    for class_ in classes:
        class_idx = class_dict[class_]
        counts[class_idx] = count_occurrence(class_, labels)
    return counts

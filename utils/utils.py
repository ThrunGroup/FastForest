import numpy as np
import itertools
from collections import defaultdict
from typing import DefaultDict, Tuple, List

from data_structures.histogram import Histogram


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


def choose_bin_type(D: int, N: int, B: int) -> str:
    """
    Return a type of bin we use depending on the number of unique feature values, data, and bins.

    :param D: Number of discrete feature vlaues
    :param N: Number of data
    :param B: Number of bins
    :return: Return one among three bin types--linear, discrete, and identity
    """
    min_num = min(D, N, B)
    if min_num == D:
        return "discrete"
    elif min_num == N:
        return "identity"
    return "linear"


def make_histograms(
    is_classification: bool,
    data: np.ndarray,
    labels: np.ndarray,
    discrete_bins_dict: DefaultDict,
    fixed_bin_type: str = "",
    num_bins: int = 11,
) -> Tuple[List[Histogram], List, List]:
    """
    Choose a bin type and number of bins, and make a histogram. Add it to histograms list. Also, filter
    extraneous bins by creating lists of considered indices and not considered indices.

    :param is_classification:  Whether is a classification problem(True) or a regression problem(False)
    :param data: A 2d-array of input data
    :param labels: An 1d-array of target dat
    :param discrete_bins_dict: A DefaultDict mapping feature index to unique feature values
    :param fixed_bin_type: Fixed type of bin which should be one of "linear", "discrete", and "identity"
    :param num_bins: Number of bins
    :return: A list of histograms, a list of indices not considered, and a list of indices considered
    """
    N = len(data)
    B = num_bins
    histograms = []
    not_considered_idcs, considered_idcs = [], []
    classes = tuple(np.unique(labels))
    for f_idx in range(len(data[0])):
        min_bin, max_bin = 0, 0
        f_data = data[:, f_idx]
        if len(discrete_bins_dict[f_idx]) == 0:
            D = float("inf")  # "f_idx"th feature isn't discrete
        else:
            D = len(discrete_bins_dict[f_idx])

        if fixed_bin_type == "":
            bin_type = choose_bin_type(D, N, B)
        else:
            bin_type = fixed_bin_type

        if bin_type == "discrete":
            num_bins = D
            assert (
                len(discrete_bins_dict[f_idx]) > 0
            ), "discrete_bins_dict[f_idx] is empty"
        elif bin_type == "identity":
            num_bins = N
        elif bin_type == "linear":
            min_bin, max_bin = np.min(f_data), np.max(f_data)
            num_bins = B
        else:
            NotImplementedError("Invalid choice of bin_type")

        histogram = Histogram(
            is_classification,
            f_idx,
            discrete_bins_dict[f_idx],
            f_data,
            classes=classes,
            num_bins=num_bins,
            min_bin=min_bin,
            max_bin=max_bin,
            bin_type=bin_type
        )
        histograms.append(histogram)

        # Filtering extraneous bins
        not_considered_idcs += list(
            itertools.product([f_idx], range(histogram.num_bins, B))
        )
        considered_idcs += list(itertools.product([f_idx], range(histogram.num_bins)))
    return histograms, not_considered_idcs, considered_idcs
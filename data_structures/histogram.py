import numpy as np
import bisect
from typing import List, Any, Tuple


class Histogram:
    """
    Histogram class that maintains a running histogram of the sampled data
    --> Should have one Histogram for each feature
    """

    def __init__(
        self,
        feature_idx: int,
        classes: Tuple[Any] = (
            0,
            1,
        ),  # classes is the tuple of labels (labels can be any type)
        num_bins: int = 11,
        min_bin: float = 0.0,
        max_bin: float = 1.0,
    ):
        self.feature_idx = feature_idx
        self.classes = classes
        self.num_bins = num_bins
        self.min_bin = min_bin
        self.max_bin = max_bin

        # TODO: Don't hardcode these edges, maybe use max and min feature values?
        # this creates middle_bins + 2 virtual bins to include tails
        self.bin_edges = np.linspace(
            min_bin, max_bin, num_bins
        )  # These are not at even numbers, just evenly spaced

        # Note: labels can be any type like string or list
        self.left = np.zeros((num_bins, len(classes)), dtype=np.int32)
        self.right = np.zeros((num_bins, len(classes)), dtype=np.int32)

    @staticmethod
    def get_bin(val: float, bin_edges: np.ndarray) -> int:
        """
        Binary Search for the value in bin_edges array.
        NOTE:
            - if value equals one of the edges, get_bin returns index + 1
            - returns len(bin_edges) + 1 if val is greater than the biggest edge which is ok
              since len(count_bucket) = len(bin_edges) + 1
        """

        return bisect.bisect_left(bin_edges, val)

    def set_bin(self, bin_array: np.ndarray):
        self.bin_edges = bin_array

    def add(self, X: np.ndarray, Y: np.ndarray):
        """
        Given dataset X , add all the points in the dataset to the histogram.
        :param X: dataset to be histogrammed (subset of original X, although could be the same size)
        :return: None, but modify the histogram to include the relevant feature values
        """
        assert (
            len(self.bin_edges)
            == np.size(self.left, axis=0)
            == np.size(self.right, axis=0)
        ), "Error: histogram is malformed"

        assert len(X) == len(Y), "Error: sample sizes and label sizes must be the same"

        feature_values = X[:, self.feature_idx]
        for idx, f in enumerate(feature_values):
            y = Y[idx]
            y_idx = self.classes.index(y)
            insert_idx = self.get_bin(val=f, bin_edges=self.bin_edges)
            # left, right[x, y] gives # of data on the left and right of xth bin of yth class
            self.right[:insert_idx, y_idx] += 1
            self.left[insert_idx:, y_idx] += 1

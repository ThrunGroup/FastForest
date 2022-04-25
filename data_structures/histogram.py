import numpy as np
import bisect
from typing import Any, Tuple


class Histogram:
    """
    Histogram class that maintains a running histogram of the sampled data
    --> Should have one Histogram for each feature
    """

    def __init__(
        self,
        feature_idx: int,
        feature_array: np.ndarray,
        data: np.ndarray,
        classes: Tuple[Any] = (
            0,
            1,
        ),  # classes is the tuple of labels (labels can be any type)
        num_bins: int = 11,
        min_bin: float = 0.0,
        max_bin: float = 1.0,
        bin_type: str = "linear",
    ):
        self.feature_idx = feature_idx
        self.feature_array = feature_array  # An array of unique feature values
        self.data = data
        self.classes = classes
        self.num_bins = num_bins
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.bin_type = bin_type

        if self.bin_type == "linear":
            self.bin_edges = self.linear_bin()
        elif self.bin_type == "discrete":
            self.bin_edges = self.discrete_bin()
        elif self.bin_type == "identity":
            self.bin_edges = self.identity_bin()
        else:
            raise NotImplementedError("Invalid type of bin")

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

    def linear_bin(self):
        return np.linspace(self.min_bin, self.max_bin, self.num_bins)

    def discrete_bin(self):
        width = int(len(self.feature_array) / self.num_bins)
        return np.array([self.feature_array[width * i] for i in range(self.num_bins)])

    def identity_bin(self):
        return np.sort(self.data)  # Copied array

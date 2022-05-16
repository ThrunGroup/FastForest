import numpy as np
import bisect
import math
from typing import Any, Tuple

from utils.constants import LINEAR, DISCRETE, IDENTITY, RANDOM, DEFAULT_NUM_BINS


class Histogram:
    """
    Histogram class that maintains a running histogram of the sampled data
    --> Should have one Histogram for each feature
    """

    def __init__(
        self,
        is_classification: bool,
        feature_idx: int,
        unique_fvals: np.ndarray,
        f_data: np.ndarray,
        classes: Tuple[Any] = (
            0,
            1,
        ),  # classes is the tuple of labels (labels can be any type)
        num_bins: int = DEFAULT_NUM_BINS,
        min_bin: float = 0.0,
        max_bin: float = 1.0,
        bin_type: str = LINEAR,
    ):
        self.feature_idx = feature_idx
        self.unique_fvals = unique_fvals
        self.f_data = f_data
        self.classes = classes
        self.num_bins = num_bins
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.bin_type = bin_type
        self.is_classification = is_classification

        if self.bin_type == LINEAR:
            self.bin_edges = self.linear_bin()
        elif self.bin_type == DISCRETE:
            self.bin_edges = self.discrete_bin()
        elif self.bin_type == IDENTITY:
            self.bin_edges = self.identity_bin()
        elif self.bin_type == RANDOM:
            self.bin_edges = self.random_bin()
        else:
            raise NotImplementedError("Invalid type of bin")

        # Note: labels can be any type like string or list
        if self.is_classification:
            self.left = np.zeros((self.num_bins, len(classes)), dtype=np.int32)
            self.right = np.zeros((self.num_bins, len(classes)), dtype=np.int32)
        else:
            # self.left_pile[i] is the list of all target values that are on the left of ith bin
            # self.right_pile[i] is the list of all target values that are on the right of ith bin
            self.left_pile = [[] for i in range(self.num_bins)]
            self.right_pile = [[] for i in range(self.num_bins)]

    def get_bin(self, val: float, bin_edges: np.ndarray) -> int:
        """
        Search for the value in bin_edges array. This operation does a O(1) search if bin_type is LINEAR,
        otherwise it does a binary search over the number of bin_edges.
        NOTE:
            - if value equals one of the edges, get_bin returns index + 1
            - returns len(bin_edges) + 1 if val is greater than the biggest edge which is ok
              since len(count_bucket) = len(bin_edges) + 1
        """
        if self.bin_type == LINEAR:
            if self.num_bins == 1 or self.min_bin == self.max_bin:
                return 0

            bin_width = (self.max_bin - self.min_bin) / (self.num_bins - 1)
            float_idx = (val - self.min_bin) / bin_width
            if float_idx < 0:
                return 0
            elif float_idx > self.num_bins - 1:
                return self.num_bins
            else:
                return math.ceil(float_idx)

        # any other bin type uses binary search
        return bisect.bisect_left(bin_edges, val)

    def set_bin(self, bin_array: np.ndarray):
        self.bin_edges = bin_array

    def empty_samples(self, bin_idcs: np.ndarray) -> None:
        """
        Empty the samples stored in bins

        :param bin_idcs: Bin indices we want to empty
        """
        if self.is_classification:
            self.left[bin_idcs, :] = np.zeros(len(self.classes), dtype=np.int64)
            self.right[bin_idcs, :] = np.zeros(len(self.classes), dtype=np.int64)
        else:
            for bin_idx in bin_idcs:
                self.left_pile[bin_idx] = []
                self.right_pile[bin_idx] = []

    def add(self, X: np.ndarray, Y: np.ndarray):
        """
        Given dataset X , add all the points in the dataset to the histogram.
        :param X: dataset to be histogrammed (subset of original X, although could be the same size)
        :return: None, but modify the histogram to include the relevant feature values
        """
        feature_values = X[:, self.feature_idx]
        if self.is_classification:
            assert (
                len(self.bin_edges)
                == np.size(self.left, axis=0)
                == np.size(self.right, axis=0)
            ), "Error: histogram is malformed"

            assert len(X) == len(
                Y
            ), "Error: sample sizes and label sizes must be the same"

            for idx, f in enumerate(feature_values):
                y = Y[idx]
                y_idx = self.classes.index(y)
                insert_idx = self.get_bin(val=f, bin_edges=self.bin_edges)
                # left, right[x, y] gives number of points on the left and right of xth bin of yth class
                self.right[:insert_idx, y_idx] += 1
                self.left[insert_idx:, y_idx] += 1

        else:
            for idx, f in enumerate(feature_values):
                y = Y[idx]
                insert_idx = self.get_bin(val=f, bin_edges=self.bin_edges)
                for right_idx in range(insert_idx):
                    self.right_pile[right_idx].append(y)
                for left_idx in range(insert_idx, self.num_bins):
                    self.left_pile[left_idx].append(y)

    def linear_bin(self) -> np.ndarray:
        """
        :return: Returns an array of bins with linear spacing
        """
        return np.linspace(self.min_bin, self.max_bin, self.num_bins)

    def discrete_bin(self) -> np.ndarray:
        """
        Returns a subset of self.feature_values with constant width. It can be either similar or different
        to linear bin depending on the distribution of self.feature_values.
        Ex) self.f_data = [0, 1, 2, 3, 3, 3, 3, 100]
            self.feature_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 100]
            self.num_bins = 5
            self.linear_bin() = [0, 25, 50, 75, 100]
            self.discrete_bin() = [0, 2, 4, 6, 8]

        :return: Return a subset array of self.feature_values
        """
        width = int(len(self.unique_fvals) / self.num_bins)
        return np.array([self.unique_fvals[width * i] for i in range(self.num_bins)])

    def identity_bin(self) -> np.ndarray:
        """
        Ex) self.f_data = [0, 1, 2, 3, 3, 3, 3, 100]
            self.identity_bin() = [0, 1, 2, 3, 100]

        :return: Return an unique sorted values array of self.f_data
        """
        identity_bin = np.unique(self.f_data)
        self.num_bins = len(identity_bin)
        return identity_bin

    def random_bin(self) -> np.ndarray:
        """
        Returns num_bins random selection of self.feature_values where min, max are
        the min, max of f_data.

        :return: Return a sorted random subset array of self.feature_values
        """
        splits = np.random.uniform(self.min_bin, self.max_bin, size=self.num_bins)
        # sorting is necessary for now since we're using binary search to find the correct bin
        return np.sort(splits, kind="mergesort")

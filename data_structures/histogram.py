import numpy as np
import bisect
import math
from typing import Any, Tuple

from utils.constants import LINEAR, DISCRETE, IDENTITY, RANDOM, DEFAULT_NUM_BINS
from utils.utils_histogram import welford_variance_calc


class Histogram:
    """
    Histogram class that maintains a running histogram of the sampled data
    --> Should have one Histogram for each feature
    """

    def __init__(
        self,
        is_classification: bool,
        feature_idx: int,
        unique_fvals: np.ndarray = None,
        f_data: np.ndarray = None,
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
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        if self.bin_type == LINEAR:
            if self.min_bin == max_bin:  # To resolve the case when self.min == self.max
                self.num_bins = 0
                return
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
            self.left = np.zeros((self.num_bins, len(classes)), dtype=np.int64)
            self.right = np.zeros((self.num_bins, len(classes)), dtype=np.int64)
        else:
            # self.left_pile[i], self.right_pile[i] stores [number of previous samples, mean of previous samples,
            # variance of previous samples] that are on the left and right of ith bin
            self.left_pile = np.zeros((self.num_bins, 3))
            self.right_pile = np.zeros((self.num_bins, 3))
            self.curr_pile = np.zeros(3)

    def get_bin(self, fvals: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
        """
        Search for the value in bin_edges array. This operation does a O(1) search if bin_type is LINEAR,
        otherwise it does a binary search over the number of bin_edges.
        NOTE:
            - if value equals one of the edges, get_bin returns index
            - returns len(bin_edges) + 1 if val is greater than the biggest edge which is ok
              since len(count_bucket) = len(bin_edges) + 1
        """
        # TODO: shouldn't be creating a histogram for feature with only one bin value
        # assert (self.max_bin > self.min_bin), "shouldn't be creating a histogram for one bin value"

        # TODO: Fix rounding error of search when bin type is linear
        if self.bin_type == LINEAR:
            bin_width = (self.max_bin - self.min_bin) / (self.num_bins - 1)
            insert_idcs = (fvals - self.min_bin) / bin_width
            insert_idcs = np.ceil(insert_idcs)
            insert_idcs[insert_idcs < 0] = 0
            insert_idcs[insert_idcs > self.num_bins - 1] = self.num_bins
            return insert_idcs

        # any other bin type uses binary search
        return np.searchsorted(bin_edges, fvals)

    def set_bin(self, bin_array: np.ndarray):
        self.bin_edges = bin_array

    def empty_samples(self, bin_idcs: np.ndarray, is_curr_empty: bool = True) -> None:
        """
        Empty the samples stored in bins

        :param bin_idcs: Bin indices we want to empty
        :param is_curr_empty: Whether to empty curr
        """
        if self.is_classification:
            self.left[bin_idcs, :] = np.zeros(len(self.classes), dtype=np.int64)
            self.right[bin_idcs, :] = np.zeros(len(self.classes), dtype=np.int64)
        else:
            self.left_pile[bin_idcs, :] = np.zeros(3)
            self.right_pile[bin_idcs, :] = np.zeros(3)
            if is_curr_empty:
                self.curr_pile = np.zeros(3)

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
                == self.num_bins
            ), "Error: histogram is malformed"

            assert len(X) == len(
                Y
            ), "Error: sample sizes and label sizes must be the same"
            insert_idcs = self.get_bin(feature_values, self.bin_edges).astype("int64")
            new_Y = self.replace_array(Y, self.class_to_idx)
            hist = np.zeros(
                (self.left.shape[0] + 1, self.left.shape[1]), dtype=np.int64
            )
            for b_idx in range(self.num_bins + 1):
                """
                 Do concatenation to include all labels when calling np.unique
                 ex) self.classes = (0,1,2)
                     new_Y[insert_idcs == b_idx] = [0,0,0,0,1,1,1,1]
                     _, counts = np.unique(new_Y, return_counts=True)
                     Then, counts = [4,4], but [4, 4, 0] is expected.
                """
                hist[b_idx] = np.bincount(
                    np.concatenate(
                        (
                            new_Y[insert_idcs == b_idx],
                            np.array(range(len(self.classes))),
                        )
                    )
                )
            hist -= 1
            for b_idx in range(self.num_bins + 1):
                self.right[:b_idx] += hist[b_idx]
                self.left[b_idx:] += hist[b_idx]
        else:
            assert (
                len(self.bin_edges)
                == np.size(self.left_pile, axis=0)
                == np.size(self.right_pile, axis=0)
                == self.num_bins
            ), "Error: histogram is malformed"
            insert_idcs = self.get_bin(feature_values, self.bin_edges)
            for b_idx in range(self.num_bins):
                self.left_pile[b_idx] = self.update_bins(
                    self.left_pile[b_idx], Y[insert_idcs <= b_idx]
                )
                self.right_pile[b_idx] = self.update_bins(
                    self.right_pile[b_idx], Y[insert_idcs > b_idx]
                )
            self.curr_pile = self.update_bins(self.curr_pile, Y)

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

    @staticmethod
    def update_bins(prev: np.ndarray, curr_data: np.ndarray):
        """
        Update bins for regression. prev contains (number of samples, mean of samples, variance of samples) that
        are previously drawn. curr_data is the array of data we newly sample.
        """
        if len(curr_data) == 0:
            return prev
        num1, mean1, var1 = prev
        num2, mean2, var2 = (
            len(curr_data),
            float(np.mean(curr_data)),
            float(np.var(curr_data)),
        )
        new_num = num1 + num2
        new_mean = (num1 * mean1 + num2 * mean2) / new_num
        new_var = welford_variance_calc(num1, mean1, var1, num2, mean2, var2)
        return new_num, new_mean, new_var

    @staticmethod
    def replace_array(array: np.ndarray, dictionary: dict):
        """
        Assert that all the elements of array should be integers.
        Apply a dictionary mapping to every elements of the array.
        And returns the new replaced array.

        Ex) dictionary = {0: 3, 1: 4)
            array = np.array([0, 0, 1, 1, 0, 0])
            replace_array(array, dictionary) gives np.array([3, 3, 4, 4, 3, 3])
        """
        # A vectorized way to replace elements, see https://bit.ly/3tk4h64.
        keys = np.array(list(dictionary.keys()))
        values = np.array(list(dictionary.values()))

        mapping_ar = np.zeros(keys.max() + 1, dtype=values.dtype)
        mapping_ar[keys] = values
        return mapping_ar[array]

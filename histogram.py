import numpy as np
import bisect


class Histogram:
    """
    Histogram class that maintains a running histogram of the sampled data
    --> Should instantiate an object for each feature_idx
    """

    def __init__(self, feature_idx: int, num_bins: int = 11, min_bin: float = .0, max_bin: float = 1.0):
        self.feature_idx = feature_idx
        self.num_bins = num_bins
        self.min_bin = min_bin
        self.max_bin = max_bin

        # TODO: Don't hardcode these edges, maybe use max and min feature values?
        # this creates middle_bins + 2 virtual bins to include tails
        self.bin_edges = np.linspace(min_bin, max_bin, num_bins)
        self.left_zeros = np.zeros(num_bins, dtype=np.int32)
        self.left_ones = np.zeros(num_bins, dtype=np.int32)
        self.right_zeros = np.zeros(num_bins, dtype=np.int32)
        self.right_ones = np.zeros(num_bins, dtype=np.int32)


    def return_decomposition(self):
        return self.left_zeros, self.left_ones, self.right_zeros, self.right_ones

    @staticmethod
    def get_bin(val: int, bin_edges: np.ndarray) -> int:
        """
        Binary Search for the value in bin_edges array.
        NOTE:
            - if value equals one of the edges, get_bin returns index + 1
            - returns len(bin_edges) + 1 if val is greater than the biggest edge which is ok
              since len(count_bucket) = len(bin_edges) + 1
        """

        return bisect.bisect_right(bin_edges, val)

    def add(self, _X: np.ndarray):
        """
        Given dataset _X , add all the points in the dataset to the histogram.
        :param _X: dataset to be histogrammed (subset of original X, although could be the same size)
        :return: None, but modify the histogram to include the relevant feature values
        """
        assert (len(self.bin_edges) == len(self.left_zeros) == len(self.left_ones)
                == len(self.right_zeros) == len(self.right_ones), "Histogram is malformed")

        feature_values = _X[:, self.feature_idx]
        Y = _X[:, -1]
        for idx, f in enumerate(feature_values):
            y = Y[idx]
            insert_idx = self.get_bin(val=f, bin_edges=self.bin_edges)
            if y == 0:
                self.right_zeros[:insert_idx] += 1
                self.left_zeros[:insert_idx] += 1
            elif y == 1:
                self.right_ones[:insert_idx] += 1
                self.left_ones[:insert_idx] += 1
            else:
                Exception(f'{i}th output of Y is not equal to 0 or 1')
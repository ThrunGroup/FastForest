import numpy as np
import bisect


class Histogram:
    """
    Histogram class that maintains a running histogram of the sampled data
    --> Should instantiate an object for each feature_idx
    """
    def __init__(self, feature_idx, middle_bins=10):
        self.feature_idx = feature_idx
        self.B = middle_bins

        # TODO: Don't hardcode these edges, maybe use max and min feature values?
        # this creates middle_bins + 2 virtual bins to include tails
        self.bin_edges = np.linspace(0.0, 1.0, self.B + 1)

        # zeros should contain the number of zeros to the left of every bin edge, except for the last element which
        # contains the number of zeros to the right of the max. Similarly for ones.
        self.zeros = np.zeros(self.B + 2, dtype=np.int32)  # + 2 for tails
        self.ones = np.zeros(self.B + 2, dtype=np.int32)  # + 2 for tails

    def return_decomposition(self):
        return (self.bin_edges, self.zeros, self.ones)

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
        assert len(self.zeros) == len(self.ones) == len(self.bin_edges) + 1, "Histogram is malformed"

        feature_values = _X[:, self.feature_idx]
        Y = _X[:, -1]
        for idx, f in enumerate(feature_values):
            y = Y[idx]
            count_bucket = self.zeros if y == 0 else self.ones
            count_bucket[self.get_bin(f, self.bin_edges)] += 1

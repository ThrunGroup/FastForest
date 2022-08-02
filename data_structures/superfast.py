import numba
from numba import njit, config
from numba.typed import List

import numpy as np
import time

from utils.solvers import solve_mab

from matplotlib import pyplot as plt

from utils.utils import get_subset_2d


@njit
def equal_den_histogram(x, num_bin):  # see https://bit.ly/3zOyBcA
    npt = len(x)
    return np.interp(np.linspace(0, npt, num_bin + 1), np.arange(npt), np.sort(x))


@njit
def convert_to_discrete(
    data: np.ndarray, num_bins: int, max_samples: int = 10000, use_quantile: bool = True
):
    n: int = data.shape[0]
    f: int = data.shape[1]
    num_bins_list = []  # New num bins
    if n > max_samples:
        sample_indices = np.random.randint(0, n, size=max_samples)
    else:
        sample_indices = np.arange(n)
    for feature_idx in range(f):
        if use_quantile:
            new_num_bin = min(
                len(np.unique(data[sample_indices, feature_idx])), num_bins
            )
            histogram = equal_den_histogram(
                data[sample_indices, feature_idx], new_num_bin
            )[1:]
            data[:, feature_idx] = np.searchsorted(histogram, data[:, feature_idx])
            num_bins_list.append(new_num_bin)
    return np.array(num_bins_list)


@njit
def get_histograms(num_bins_list: np.ndarray) -> List[np.ndarray]:
    """
    :param num_bins_list: 1d array of number of bins
    """
    histograms = []
    NUM_CLASSES = 2
    for num_bin in num_bins_list:
        histograms.append(
            np.zeros((num_bin, NUM_CLASSES))
        )  # Hard-coded for only binary case
    return histograms


@njit
def reset_histograms(histograms: List[np.ndarray]):
    for idx in range(len(histograms)):
        histograms[idx] = np.zeros_like(histograms[idx])


@njit
def find_mab_split(
    data: np.ndarray,  # original data
    labels: np.ndarray,  # original labels
    histograms: List[np.ndarray],
    candidates: np.ndarray,  # Regard features as arms
    indices: np.ndarray,
    start: int,
    end: int,
    batch_size: int,
):
    NUM_CLASSES = 2
    time_step = 0
    while len(candidates) > 1:
        impurity_array = np.empty(len(candidates))
        cb_deltas = np.empty(len(candidates))
        min_idcs = np.empty(len(candidates))
        time_step += 1
        sample_indices = np.random.randint(indices[start], indices[end] + 1, batch_size)
        for candidate_idx in range(len(candidates)):
            feature = candidates[candidate_idx]
            histogram = histograms[feature]
            for sample_idx in sample_indices:
                histogram[data[sample_idx, feature], labels[sample_idx]] += 1
            counts_left = np.zeros_like(histogram)
            counts_right = np.zeros_like(histogram)
            counts_left[0] = histogram[0]
            for j in range(1, histogram.shape[0]):
                counts_left[j] += counts_left[j - 1] + histogram[j]
                counts_right[histogram.shape[0] - 1 - j] = (
                    counts_right[histogram.shape[0] - j]
                    + histogram[histogram.shape[0] - j]
                )
            n_left = np.expand_dims(np.sum(counts_left, axis=1), axis=1)
            n_right = np.expand_dims(np.sum(counts_right, axis=1), axis=1)
            left_weight = n_left / (n_right + n_left)
            right_weight = n_right / (n_right + n_left)

            p_left = counts_left / n_left
            p_right = counts_right / n_right

            for idx in range(len(n_left)):
                if n_left[idx] == 0:
                    p_left[idx] = np.zeros(2)
                if n_right[idx] == 0:
                    p_right[idx] = np.zeros(2)
            curr_gini = (
                1
                - np.sum((counts_left[0, :] + counts_right[0, :]) ** 2)
                / (n_left[0] + n_right[0]) ** 2
            )
            gini_vec = -(
                curr_gini
                - left_weight * np.expand_dims((1 - np.sum(p_left ** 2, axis=1)), axis=1)
                - right_weight * np.expand_dims((1 - np.sum(p_right ** 2, axis=1)), axis=1)
            )
            min_idx = gini_vec.argmin()
            min_idcs[candidate_idx] = min_idx

            # Find the variance of best bind
            cb_delta = np.sum(
                (
                    (left_weight[min_idx] ** 2)  # weight
                    * (
                        (2 * p_left[min_idx, :-1] * (2 * p_left[min_idx, -1] - 1)) ** 2
                    )  # dG/dp ** 2
                    * (
                        p_left[min_idx] * (1 - p_left[min_idx]) / n_left[min_idx]
                    )  # V_p ** 2
                )
                + (
                    (right_weight[min_idx] ** 2)  # weight
                    * (
                        (2 * p_right[min_idx, :-1] * (2 * p_right[min_idx, -1] - 1))
                        ** 2
                    )  # dG/dp ** 2
                    * (
                        (p_right[min_idx] * (1 - p_right[min_idx]) / n_right[min_idx])
                        ** 2
                    )  # V_p ** 2
                )
            )

            # Update
            impurity_array[candidate_idx] = gini_vec[min_idx, 0]
            cb_deltas[candidate_idx] = np.sqrt(cb_delta)
        ucbs = impurity_array + cb_deltas * 3
        lcbs = impurity_array - cb_deltas * 3
        surviving_candidates = []
        min_ucbs = ucbs.min()
        min_impurity = impurity_array.min()
        for idx in range(len(candidates)):
            if (
                (lcbs[idx] <= min_ucbs)
                and (lcbs[idx] < 0)
                # and (
                #     impurity_array[idx] <= 0.99 * min_impurity
                #     or impurity_array[idx] == min_impurity
                # )
            ):
                surviving_candidates.append(candidates[idx])
        candidates = np.array(surviving_candidates)
        if time_step >= data.shape[0] / batch_size:
            break
    print(candidates)
    return candidates[0], min_idcs[0], time_step, cb_deltas, impurity_array


def get_gini(counts: np.ndarray):
    return 1 - (counts / counts.sum(axis=1, keepdims=True)) ** 2


if __name__ == "__main__":
    """
    TEST!!!
    """
    config.DISABLE_JIT = False
    data = np.random.randint(low=0, high=1000, size=(1000000, 30))
    label = np.random.randint(low=0, high=2, size=1000000)
    bins_list = convert_to_discrete(data, 10)
    histograms = numba.typed.List(get_histograms(bins_list))
    indices = np.arange(data.shape[0])
    candidates = np.arange(data.shape[1])
    start = 0
    end = data.shape[0] - 1
    batch_size = 1000
    find_mab_split(
        data=data,
        labels=label,
        histograms=histograms,
        indices=indices,
        candidates=candidates,
        start=start,
        end=end,
        batch_size=batch_size,
    )

    a = time.time()
    print(find_mab_split(
        data=data,
        labels=label,
        histograms=histograms,
        indices=indices,
        candidates=candidates,
        start=start,
        end=end,
        batch_size=batch_size,
    ))
    print(time.time() - a)

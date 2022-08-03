import numba
from numba import njit, config
from numba.typed import List

import numpy as np
import time




@njit
def equal_den_histogram(x, num_bin):  # see https://bit.ly/3zOyBcA
    npt = len(x)
    return np.interp(np.linspace(0, npt, num_bin + 1), np.arange(npt), np.sort(x))

@njit
def convert_to_discrete(
    data:np.ndarray, num_bins: int, max_samples: int = 10000, use_quantile: bool = True
):
    # print(numba.typeof(data[:, 0]))
    n: int = data.shape[0]
    f: int = data.shape[1]
    num_bins_list = []  # New num bins
    new_data = np.empty(shape=data.shape, dtype=np.int8)
    if n > max_samples:
        sample_indices = np.random.randint(0, n, size=max_samples)
    else:
        sample_indices = np.arange(n)
    for feature_idx in range(f):
        if use_quantile:
            unique_samples = np.unique(data[sample_indices, feature_idx])
            if len(unique_samples) <= num_bins:
                histogram = unique_samples
                new_num_bin = len(unique_samples)
            else:
                histogram = equal_den_histogram(
                    data[sample_indices, feature_idx], num_bins
                )[1:]
                new_num_bin = num_bins
            # print(numba.typeof(data))
            a = data[:, feature_idx]
            new_data[:, feature_idx] = np.searchsorted(histogram, data[:, feature_idx]).astype(np.int8)
            num_bins_list.append(new_num_bin)
    return new_data, np.array(num_bins_list)


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
        sample_indices = indices[np.random.randint(start, end + 1, batch_size)]
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
                - left_weight
                * np.expand_dims((1 - np.sum(p_left ** 2, axis=1)), axis=1)
                - right_weight
                * np.expand_dims((1 - np.sum(p_right ** 2, axis=1)), axis=1)
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
        ucbs = impurity_array + cb_deltas * 10
        lcbs = impurity_array - cb_deltas * 10
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
    if min_impurity == 0:
        print("No best impurity exists")
        return None, None, None
    print("This is the result of mab: ", min_impurity, candidates, time_step)
    return int(candidates[0]), int(min_idcs[0]), curr_gini


def get_gini(counts: np.ndarray):
    return 1 - (counts / counts.sum(axis=1, keepdims=True)) ** 2

@njit
def splt_node_helper(
    data: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    split_feature: np.ndarray,
    split_value: np.ndarray,
):
    left_idx = start
    right_idx = end
    while left_idx != right_idx:
        if data[indices[left_idx], split_feature] <= split_value:
            left_idx += 1
        else:
            indices[left_idx], indices[right_idx] = (
                indices[right_idx],
                indices[left_idx],
            )
            right_idx -= 1
    left_start = start
    right_end = end
    if data[indices[left_idx], split_feature] <= split_value:
        left_end = left_idx
        right_start = left_idx + 1
    else:
        left_end = left_idx - 1
        right_start = left_idx

    return left_start, left_end, right_start, right_end


class Node(object):
    def __init__(
        self, data, labels, indices, start, end, histograms, depth, max_features,
    ):
        self.data = data
        self.labels = labels
        self.indices = indices
        self.start = start
        self.end = end
        self.histograms = histograms
        self.depth = depth

        self.left_child = None
        self.right_child = None
        self.max_features = max_features
        self.features = np.arange(data.shape[1])
        self.features = np.random.choice(
            data.shape[1], int(max_features * data.shape[1]), replace=False
        )

        self.split_feature = None
        self.split_value = None
        self.curr_impurity = None
        self.find_best_threshold = False

    def mab_split(self):
        reset_histograms(self.histograms)
        if not self.find_best_threshold:
            self.split_feature, self.split_value, self.curr_impurity = find_mab_split(
                data=data,
                labels=labels,
                indices=indices,
                start=start,
                end=end,
                histograms=histograms,
                candidates=self.features,
                batch_size=min(30, self.data.shape[0] / 1000),
            )
            print(self.split_feature, self.split_value, self.curr_impurity)
            self.find_best_threshold = True

    def split(self):
        left_start, left_end, right_start, right_end = splt_node_helper(
            data=self.data,
            labels=self.labels,
            indices=self.indices,
            start=self.start,
            end=self.end,
            split_feature=self.split_feature,
            split_value=self.split_value,
        )
        self.left_child = Node(
            data=self.data,
            labels=self.labels,
            indices=self.indices,
            start=left_start,
            end=left_end,
            histograms=self.histograms,
            depth=self.depth + 1,
            max_features=self.max_features,
        )
        self.right_child = Node(
            data=self.data,
            labels=self.labels,
            indices=self.indices,
            start=right_start,
            end=right_end,
            histograms=self.histograms,
            depth=self.depth + 1,
            max_features=self.max_features,
        )
        return self.left_child, self.right_child



if __name__ == "__main__":
    """
    TEST!!!
    """
    ## Compiling ##
    # Jay: need to fix bug when type is integer
    data = np.random.random(size=(10,3)).astype(np.float32)
    a = np.copy(data)
    labels = np.random.randint(low=0, high=2, size=10).astype(np.int8)
    data, num_bins_list = convert_to_discrete(data, 10)
    histograms = numba.typed.List(get_histograms(num_bins_list))
    indices = np.arange(data.shape[0])
    candidates = np.arange(data.shape[1])
    start = 0
    end = data.shape[0] - 1
    batch_size = 3
    print(data.dtype)
    find_mab_split(
        data=data,
        labels=labels,
        histograms=histograms,
        indices=indices,
        candidates=candidates,
        start=start,
        end=end,
        batch_size=batch_size,
    )

    config.DISABLE_JIT = False
    data = np.random.random(size=(1000000, 30)).astype(np.float32)
    # labels = np.random.randint(low=0, high=2, size=1000000)
    labels = np.random.randint(low=0, high=2, size=1000000)
    a = time.time()
    data, num_bins_list = convert_to_discrete(data, 10)
    histograms = numba.typed.List(get_histograms(num_bins_list))
    indices = np.arange(data.shape[0])
    candidates = np.arange(data.shape[1])
    start = 0
    end = data.shape[0] - 1
    batch_size = 1000
    print(
        find_mab_split(
            data=data,
            labels=labels,
            histograms=histograms,
            indices=indices,
            candidates=candidates,
            start=start,
            end=end,
            batch_size=batch_size,
        )
    )
    # print(time.time() - a)
    # a = time.time()
    # print(solve_mab(data_2, labels))
    # print(time.time() - a)

    root_node = Node(
        data=data,
        labels=labels,
        indices=indices,
        start=start,
        end=end,
        histograms=histograms,
        depth=0,
        max_features=1,
    )
    a = time.time()
    root_node.mab_split()
    root_node.split()
    print(time.time() - a)

    root_node = Node(
        data=data,
        labels=labels,
        indices=np.arange(data.shape[0]),
        start=start,
        end=end,
        histograms=histograms,
        depth=0,
        max_features=1,
    )
    a = time.time()
    root_node.mab_split()
    root_node.split()
    print(time.time() - a)
import numba
import sklearn.tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from numba import njit, config
from numba.typed import List
from typing import Union, Tuple
import numpy as np
import time

from experiments.datasets.data_loader import fetch_data, get_large_flight_data
from utils.constants import FLIGHT, COVTYPE, APS, MNIST_STR
from data_structures.tree_classifier import TreeClassifier


def r(num):
    return round(num, 3)


@njit
def equal_den_histogram(x, num_bin):  # see https://bit.ly/3zOyBcA
    npt = len(x)
    return np.interp(
        np.linspace(0, npt, num_bin + 1), np.arange(npt), np.sort(x)
    ).astype(x.dtype)


@njit
def convert_to_discrete(
    data: np.ndarray,
    num_bins: int,
    max_samples: int = 100000,
    use_quantile: bool = True,
):
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
                bin_edges = unique_samples
                new_num_bin = len(unique_samples) + 1
            else:
                bin_edges = equal_den_histogram(
                    data[sample_indices, feature_idx], num_bins
                )[1:-1]
                new_num_bin = num_bins
            new_data[:, feature_idx] = np.searchsorted(
                bin_edges, data[:, feature_idx]
            ).astype(np.int8)
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
    min_impurity_decrease: float = 0,
):
    NUM_CLASSES = 2
    time_step = 0
    while len(candidates) > 1:
        impurity_array = np.empty(len(candidates))
        cb_deltas = np.empty(len(candidates))
        min_bin_idcs = np.empty(len(candidates))
        time_step += 1
        if batch_size >= (end - start + 1):
            sample_indices = indices[start : end + 1]
        else:
            sample_indices = indices[np.random.randint(start, end + 1, batch_size)]
        for candidate_idx in range(len(candidates)):
            feature = candidates[candidate_idx]
            histogram = histograms[feature]
            for sample_idx in sample_indices:
                histogram[data[sample_idx, feature], labels[sample_idx]] += 1
            counts_left = np.zeros(
                (histogram.shape[0] - 1, histogram.shape[1])
            )  # Split point = # of bins - 1
            counts_right = np.zeros((histogram.shape[0] - 1, histogram.shape[1]))
            counts_left[0] = histogram[0]
            counts_right[-1] = histogram[-1]
            for j in range(1, counts_left.shape[0]):
                counts_left[j] += counts_left[j - 1] + histogram[j]
                counts_right[counts_left.shape[0] - 1 - j] = (
                    counts_right[counts_left.shape[0] - j]
                    + histogram[counts_left.shape[0] - j]
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
            min_bin_idcs[candidate_idx] = min_idx

            # Find the variance of best bind
            # Jay: This is variance for binary classification. Have to fix this
            cb_delta = np.sum(
                (
                    (left_weight[min_idx] ** 2)  # weight
                    * (
                        (2 * p_left[min_idx, :-1] * (2 * p_left[min_idx, -1] - 1)) ** 2
                    )  # dG/dp ** 2
                    * (
                        p_left[min_idx, 0] * (1 - p_left[min_idx, 0]) / n_left[min_idx]
                    )  # V_p ** 2
                )
                + (
                    (right_weight[min_idx] ** 2)  # weight
                    * (
                        (2 * p_right[min_idx, :-1] * (2 * p_right[min_idx, -1] - 1))
                        ** 2
                    )  # dG/dp ** 2
                    * (
                        (
                            p_right[min_idx, 0]
                            * (1 - p_right[min_idx, 0])
                            / n_right[min_idx]
                        )
                        ** 2
                    )  # V_p ** 2
                )
            )

            # Update
            impurity_array[candidate_idx] = gini_vec[min_idx, 0]
            cb_deltas[candidate_idx] = np.sqrt(cb_delta)
        ucbs = impurity_array + cb_deltas * 2
        lcbs = impurity_array - cb_deltas * 2

        survived_candidates_indices = []
        min_ucbs = np.nanmin(ucbs)
        min_candidate_idx = impurity_array.argmin()
        min_impurity = impurity_array[min_candidate_idx]
        min_candidate = candidates[min_candidate_idx]
        for idx in range(len(candidates)):
            if (
                (lcbs[idx] <= min_ucbs)
                and (lcbs[idx] < 0)
                # and (
                #     impurity_array[idx] <= 0.99 * min_impurity
                #     or impurity_array[idx] == min_impurity
                # )
            ):
                survived_candidates_indices.append(idx)
        candidates = candidates[np.array(survived_candidates_indices)]
        if time_step >= (end - start + 1) / batch_size:
            break
    # if time_step * batch_size <= data.shape[0] / 10:
    #     new_batch_size = data.shape[0] / 10 * -time_step * batch_size
    #     feature = min_candidate
    #     # sample_indices = indices[np.random.randint(start, end + 1, new_batch_size)]
    #     sample_indices = indices[start: end + 1]
    #     histogram = histograms[feature]
    #     for sample_idx in sample_indices:
    #         histogram[data[sample_idx, feature], labels[sample_idx]] += 1
    #     counts_left = np.zeros_like(histogram)
    #     counts_right = np.zeros_like(histogram)
    #     counts_left[0] = histogram[0]
    #     for j in range(1, histogram.shape[0]):
    #         counts_left[j] += counts_left[j - 1] + histogram[j]
    #         counts_right[histogram.shape[0] - 1 - j] = (
    #             counts_right[histogram.shape[0] - j] + histogram[histogram.shape[0] - j]
    #         )
    #     n_left = np.expand_dims(np.sum(counts_left, axis=1), axis=1)
    #     n_right = np.expand_dims(np.sum(counts_right, axis=1), axis=1)
    #     left_weight = n_left / (n_right + n_left)
    #     right_weight = n_right / (n_right + n_left)
    #
    #     p_left = counts_left / n_left
    #     p_right = counts_right / n_right
    #
    #     for idx in range(len(n_left)):
    #         if n_left[idx] == 0:
    #             p_left[idx] = np.zeros(2)
    #         if n_right[idx] == 0:
    #             p_right[idx] = np.zeros(2)
    #     curr_gini = (
    #         1
    #         - np.sum((counts_left[0, :] + counts_right[0, :]) ** 2)
    #         / (n_left[0] + n_right[0]) ** 2
    #     )
    #     gini_vec = -(
    #         curr_gini
    #         - left_weight * np.expand_dims((1 - np.sum(p_left ** 2, axis=1)), axis=1)
    #         - right_weight * np.expand_dims((1 - np.sum(p_right ** 2, axis=1)), axis=1)
    #     )
    #     min_idx = gini_vec.argmin()
    #     if gini_vec[min_idx] >= min_impurity_decrease:
    #         return None, None, None
    #     return feature, min_idx, curr_gini
    if min_impurity >= min_impurity_decrease:
        # print("No best impurity exists")
        return None, None, None
    # print("This is the result of mab: ", min_impurity, candidates, time_step)
    if int(candidates[0]) < 0:
        print("what?")
    # print(candidates, min_idcs, curr_gini, min_impurity, time_step)
    return int(min_candidate), int(min_bin_idcs[min_candidate_idx]), curr_gini


def get_gini(counts: np.ndarray):
    return 1 - (counts / counts.sum(axis=1, keepdims=True)) ** 2


class Node(object):
    def __init__(
        self,
        data,
        labels,
        indices,
        start: int,
        end: int,
        histograms,
        depth,
        max_features,
        batch_size=None,
    ):
        self.data = data
        self.labels = labels
        self.indices = indices
        self.start = int(start)
        self.end = int(end)
        self.histograms = histograms
        self.depth = int(depth)
        self.batch_size = batch_size

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

    def find_best_split(self):
        reset_histograms(self.histograms)
        if not self.find_best_threshold:
            self.split_feature, self.split_value, self.curr_impurity = find_mab_split(
                data=self.data,
                labels=self.labels,
                indices=self.indices,
                start=self.start,
                end=self.end,
                histograms=self.histograms,
                candidates=self.features,
                batch_size=(self.end - self.start + 1)
                if self.batch_size is None
                else self.batch_size,
            )
            self.find_best_threshold = True

    def split(self):
        if (self.find_best_threshold is False) or (
            self.split_feature is None
        ):  # Impurity reduction >= min_impurity_reduction
            return None, None
        left_start, left_end, right_start, right_end = split_node_helper(
            data=self.data,
            labels=self.labels,
            indices=self.indices,
            start=self.start,
            end=self.end,
            split_feature=self.split_feature,
            split_value=self.split_value,
        )
        if left_start is None:
            self.split_feature, self.split_value = None, None
            return None, None
        self.left_child = Node(
            data=self.data,
            labels=self.labels,
            indices=self.indices,
            start=left_start,
            end=left_end,
            histograms=self.histograms,
            depth=self.depth + 1,
            max_features=self.max_features,
            batch_size=self.batch_size,
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
            batch_size=self.batch_size,
        )
        return self.left_child, self.right_child

    def n_print(self) -> None:
        """
        Print the node's children depth-first
        Me: split x < 5:
        """
        assert (self.left_child and self.right_child) or (
            self.left_child is None and self.right_child is None
        ), "Error: split is malformed"
        if self.left_child:
            print(
                ("|   " * self.depth)
                + "|--- feature_"
                + str(self.split_feature)
                + " <= "
                + str(self.split_value)
            )
            self.left_child.n_print()
            print(
                ("|   " * self.depth)
                + "|--- feature_"
                + str(self.split_feature)
                + " > "
                + str(self.split_value)
            )
            self.right_child.n_print()
        else:
            class_idx_pred = np.argmax(
                np.bincount(self.labels[self.indices[self.start : self.end + 1]])
            )
            print(("|   " * self.depth) + "|--- " + "class: " + str(class_idx_pred))


class Tree:
    def __init__(
        self,
        data,
        labels,
        indices,
        start,
        end,
        histograms,
        max_depth,
        max_features,
        batch_size=None,
    ):
        self.data = data
        self.labels = labels
        self.indices = indices
        self.start = start
        self.end = end
        self.histograms = histograms
        self.max_depth = max_depth
        self.max_features = max_features
        self.batch_size = batch_size

        self.record = None
        self.num_nodes = 1
        self.is_record = False

        reset_histograms(self.histograms)
        self.node = Node(
            data=self.data,
            labels=self.labels,
            indices=self.indices,
            start=start,
            end=end,
            histograms=self.histograms,
            depth=0,
            max_features=max_features,
            batch_size=self.batch_size,
        )

    def recursive_split(self, node: Node) -> None:
        """
        Recursively split nodes till the termination condition is satisfied

        :param node: A root node to be split recursively
        """
        if node.depth < self.max_depth:
            node.find_best_split()
            node.split()
        if node.left_child is not None:
            self.recursive_split(node.left_child)
            self.recursive_split(node.right_child)
            self.num_nodes += 2

    def fit(self):
        self.recursive_split(self.node)

    def write_record_recursive(self, node: Node, record: np.ndarray, idx: int):
        record[idx] = (
            node.start,
            node.end,
            node.split_feature,
            node.split_value,
            np.nan,
        )
        if node.left_child is not None:
            assert node.right_child is not None, "Malformed tree"
            self.write_record_recursive(node.left_child, record, 2 * idx + 1)
            self.write_record_recursive(node.right_child, record, 2 * idx + 2)

    def get_record(self):
        self.record = np.empty((2 ** (self.max_depth + 1) - 1, 5))
        self.record[:] = np.nan
        self.write_record_recursive(self.node, self.record, 0)
        self.is_record = True

    def predict(self, datapoint: np.ndarray):
        assert len(datapoint.shape) == 1, "Invalid dimension"
        assert self.is_record, "Should write tree record first"
        return _predict(datapoint, self.record, self.labels, self.indices)

    def predict_batch(self, data: np.ndarray):
        assert len(data.shape) == 2, "Invalid choice of dimension"
        assert self.is_record, "Should write tree record first"
        return _predict_batch(data, self.record, self.labels, self.indices)


"""
Njit functions for Tree class and Node class
"""

# Node class helper
@njit
def split_node_helper(
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
    if left_start > left_end or right_start > right_end:
        return None, None, None, None
    return left_start, left_end, right_start, right_end


# Tree class helpers
@njit
def _predict_batch(
    data: np.ndarray, record: np.ndarray, labels: np.ndarray, indices: np.ndarray
):
    predicted_labels = np.empty(data.shape[0])
    for idx in range(data.shape[0]):
        predicted_labels[idx] = _predict(data[idx], record, labels, indices)
    return predicted_labels


@njit
def _predict(
    datapoint: np.ndarray, record: np.ndarray, labels: np.ndarray, indices: np.ndarray
):
    """
    Classifier: calculate the predicted label

    :param datapoint: datapoint to fit
    :param record: stores (start, end, split feature, split value) of each node / ith element is ith node (
    start from root and traverse from left to right and increase the depth by 1)
    :return: the probabilities of the datapoint being each class label or the mean value of labels
    """
    depth = 0
    idx = 0
    node_record = record[idx]
    while not np.isnan(node_record[2]):  # means that node is not splitted
        feature_value = datapoint[int(node_record[2])]
        if feature_value <= node_record[3]:
            idx = 2 * idx + 1
        else:
            idx = 2 * idx + 2
        node_record = record[idx]
    start = int(node_record[0])
    end = int(node_record[1])
    if np.isnan(node_record[4]):
        node_record[4] = np.bincount(labels[indices[start : end + 1]]).argmax()
    return node_record[4]


def compiling_jit(data_dtype, labels_dtype):
    print(
        "Compiling functions with njit decorator by running them with right type of parameters beforehand..."
    )
    start_time = time.time()
    data = np.random.random(size=(300, 3)).astype(data_dtype)
    labels = (data[:, 1] < 0.3).astype(labels_dtype)
    data, num_bins_list = convert_to_discrete(data, 10)
    histograms = numba.typed.List(get_histograms(num_bins_list))
    indices = np.arange(data.shape[0])
    candidates = np.arange(data.shape[1])
    start = 0
    end = data.shape[0] - 1
    batch_size = 9
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
    tree = Tree(
        data=data,
        labels=labels,
        indices=indices,
        start=start,
        end=end,
        histograms=histograms,
        max_depth=5,
        max_features=1,
        batch_size=10,
    )
    tree.get_record()
    a = time.time()
    tree.fit()
    tree.predict_batch(data)
    print(time.time() - start_time)
    print("Compile ends\n")


def run_comparison(
    data,
    labels,
    is_exact: bool = True,
    is_mab: bool = True,
    is_sklearn: bool = True,
    is_prev_mab: bool = True,
    max_depth: int = 8,
    num_experiments: int = 5,
    dataset: str = "",
):
    print(f"Starts Comparison Experiments (dataset: {dataset}, max depth = {max_depth})\n")
    pm = " \u00B1 "
    NUM_BINS = 30  # Hard-coded
    BATCH_SIZE = 1000

    start_time = time.time()
    histogrammed_data, num_bins_list = convert_to_discrete(data, NUM_BINS)
    histograms = numba.typed.List(get_histograms(num_bins_list))
    start = 0
    end = histogrammed_data.shape[0] - 1
    print(f"---Histogramming (preprocess): {time.time() - start_time} (s)")

    def run_numba_tree(batch_size, verbose: bool = True):
        tree = Tree(
            data=histogrammed_data,
            labels=labels,
            indices=np.arange(histogrammed_data.shape[0]),
            start=start,
            end=end,
            histograms=histograms,
            max_depth=max_depth,
            max_features=1.0,
            batch_size=batch_size,
        )
        start_time = time.time()
        tree.fit()
        fit_time = time.time() - start_time
        tree.get_record()
        accuracy = (
            np.sum(tree.predict_batch(histogrammed_data) == labels) / len(labels) * 100
        )
        num_nodes = tree.num_nodes
        if verbose:
            print(f"---fit time: {fit_time}s")
            print(f"---accuracy: {accuracy}%")
            print(f"---num_nodes: {tree.num_nodes}")
        return fit_time, accuracy, num_nodes

    def print_results(fit_time_list, accuracy_list, num_nodes_list):
        print(
            f"---fit time: {r(np.mean(fit_time_list))}"
            + pm
            + f"{r(np.std(fit_time_list) / np.sqrt(num_experiments))} (s)"
        )
        print(
            f"---accuracy: {r(np.mean(accuracy_list))}"
            + pm
            + f"{r(np.std(accuracy_list) / np.sqrt(num_experiments))} (%)"
        )
        print(
            f"---num_nodes: {r(np.mean(num_nodes_list))}"
            + pm
            + f"{r(np.std(num_nodes_list) / np.sqrt(num_experiments))}"
        )

    if is_exact:
        print("\nHistogrammed decision tree without MAB")
        fit_time_list = []
        accuracy_list = []
        num_nodes_list = []
        for _ in range(num_experiments):
            fit_time, accuracy, num_nodes = run_numba_tree(None, False)
            fit_time_list.append(fit_time)
            accuracy_list.append(accuracy)
            num_nodes_list.append(num_nodes)
        print_results(fit_time_list, accuracy_list, num_nodes_list)
    if is_mab:
        print("\nHistogrammed decision tree with MAB")
        fit_time_list = []
        accuracy_list = []
        num_nodes_list = []
        for _ in range(num_experiments):
            fit_time, accuracy, num_nodes = run_numba_tree(BATCH_SIZE, False)
            fit_time_list.append(fit_time)
            accuracy_list.append(accuracy)
            num_nodes_list.append(num_nodes)
        print(f"---batch_size: {BATCH_SIZE}")
        print_results(fit_time_list, accuracy_list, num_nodes_list)
    if is_sklearn:
        print("\nSklearn decision tree")
        fit_time_list = []
        accuracy_list = []
        num_nodes_list = []
        for _ in range(num_experiments):
            tree = DecisionTreeClassifier(max_depth=max_depth, max_features=None,)
            start_time = time.time()
            tree.fit(data, labels)
            fit_time_list.append(time.time() - start_time)
            accuracy_list.append(np.sum((tree.predict(data) == labels) / data.shape[0] * 100))
            num_nodes_list.append(tree.tree_.node_count)
        print_results(fit_time_list, accuracy_list, num_nodes_list)

    if is_prev_mab:
        print("\nOur previous mab decision tree")
        fit_time_list = []
        accuracy_list = []
        num_nodes_list = []
        for _ in range(num_experiments):
            tree = TreeClassifier(
                data=data, labels=labels, classes={0: 0, 1: 1}, max_depth=max_depth
            )
            start_time = time.time()
            tree.fit()
            fit_time_list.append(time.time() - start_time)
            accuracy_list.append(
                np.sum((tree.predict_batch(data)[0] == labels) / data.shape[0] * 100)
            )
            num_nodes_list.append(2 * tree.num_splits + 1)
        print_results(fit_time_list, accuracy_list, num_nodes_list)


if __name__ == "__main__":
    """
    TEST!!!
    """
    # data, labels = make_classification(
    #     500000, n_features=50, n_informative=5, random_state=0
    # )
    # labels.astype(np.int8)
    # data_dtype, labels_dtype = data.dtype, labels.dtype
    # compiling_jit(data_dtype, labels_dtype)
    # run_comparison(data=data, labels=labels, max_depth=3, dataset="Random Classification(500k)")
    # run_comparison(data=data, labels=labels, max_depth=8, dataset="Random Classification(500k)")
    #
    # data, labels, _, _ = fetch_data(FLIGHT)
    # labels.astype(np.int8)
    # data_dtype, labels_dtype = data.dtype, labels.dtype
    # compiling_jit(data_dtype, labels_dtype)
    # run_comparison(data=data, labels=labels, is_prev_mab=False, max_depth=3, dataset="Flight delay(100k)")
    # run_comparison(data=data, labels=labels, is_prev_mab=False, max_depth=8, dataset="Flight delay(100k)")
    #
    # data, labels, _, _ = get_large_flight_data()
    # labels.astype(np.int8)
    # data_dtype, labels_dtype = data.dtype, labels.dtype
    # compiling_jit(data_dtype, labels_dtype)
    # run_comparison(data=data, labels=labels, is_prev_mab=False, max_depth=3, dataset="Flight delay(1m)")
    # run_comparison(data=data, labels=labels, is_prev_mab=False, max_depth=8, dataset="Flight delay(1m)")
    #
    # data, labels, _, _ = fetch_data(COVTYPE)
    # labels.astype(np.int8)
    # data_dtype, labels_dtype = data.dtype, labels.dtype
    # compiling_jit(data_dtype, labels_dtype)
    # run_comparison(data=data, labels=labels, is_prev_mab=False, max_depth=3, dataset="Covtype")
    # run_comparison(data=data, labels=labels, is_prev_mab=False, max_depth=8, dataset="Covtype")
    #
    # data, labels, _, _ = fetch_data(APS)
    # labels.astype(np.int8)
    # data_dtype, labels_dtype = data.dtype, labels.dtype
    # compiling_jit(data_dtype, labels_dtype)
    # run_comparison(data=data, labels=labels, is_prev_mab=False, max_depth=3, dataset="APS")
    # run_comparison(data=data, labels=labels, is_prev_mab=False, max_depth=8, dataset="APS")

    data, labels, _, _ = fetch_data(MNIST_STR)
    labels = np.where(labels <= 4, 0, 1).astype(np.int8)
    labels.astype(np.int8)
    data_dtype, labels_dtype = data.dtype, labels.dtype
    compiling_jit(data_dtype, labels_dtype)
    run_comparison(data=data, labels=labels, is_prev_mab=False, max_depth=3, dataset="MNIST")
    run_comparison(data=data, labels=labels, is_prev_mab=False, max_depth=8, dataset="MNIST")
    exit()
import sys
import numpy as np
import itertools
import math
import bisect
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from typing import List, Tuple, Callable, Union
from histogram import Histogram
from collections import defaultdict


# TODO: Define histogram class
def sample_targets(X: np.ndarray, accesses: Tuple[np.ndarray, np.ndarray],
                   histograms: List[object], batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a dataset and set of features, draw batch_size new datapoints (with replacement) from the dataset. Insert
    their feature values into the (potentially non-empty) histograms and recompute the changes in impurity
    for each potential bin split
    :param X: original dataset
    :param accesses: a parameter "accesses" is Tuple(f, b) where f and b are arrays that
    (f[i], b[i]) is the position of accesses we want to update their impurity reduction. #Jay: Better to call it arms?
    :param histograms: list of the histograms for ALL feature indices
    :param batch_size: the number of samples we're going to choose

    return: impurity_reduction and its variance of accesses
    """

    # TODO(@motiwari): Samples all bin edges for a given feature, should only sample those under consideration.
    feature_idcs, bin_edge_idcs = accesses
    l = len(bin_edge_idcs)  # l is the total number of accesses we want to update
    f2bin_dict = defaultdict(list)  # f2bin_dict[i] contains bin indices list of ith feature

    for idx in range(l):  # NOTE: Using list instead of dictionary is a way more efficient, but for readability, I use
        # dictioanry here
        feature = feature_idcs[idx]
        bin_edge = bin_edge_idcs[idx]
        f2bin_dict[feature].append(bin_edge)

    # NOTE: impurity_reductions and cb_deltas are smaller subsets than the original
    impurity_reductions = np.array([], dtype=float)
    cb_deltas = np.array([], dtype=float)
    N = len(X)
    sample_idcs = np.random.choice(N, size=batch_size)  # Default: with replacement (replace=True)
    samples = X[sample_idcs]
    for f_idx, f in enumerate(f2bin_dict.keys()):
        h = histograms[f]
        h.add(samples)  # This is where the labels are used
        # TODO(@motiwari): Can make this more efficient because a lot of histogram computation is reused across steps
        i_r, cb_d = get_impurity_reductions(h, f2bin_dict[f], ret_vars=True)
        impurity_reductions = np.concatenate([impurity_reductions, i_r])
        cb_deltas = np.concatenate([cb_deltas, np.sqrt(cb_d)])  # The above fn returns the vars

    # TODO(@motiwari): This seems dangerous, because access appears to be a linear index to the array
    return impurity_reductions, cb_deltas


def solve_mab(X: np.ndarray, feature_idcs: List[int]) -> Tuple[int, float]:
    """
    Solve a multi-armed bandit problem. The objective is to find the best feature to split on, as well as the value
    that feature should be split at.

    - The arms correspond to the (feature, feature_value) pairs. There are F x B of them.
    - The true arm return corresponds to the actual reduction in impurity if we were to perform that split.
    - Pulling an arm corresponds to drawing a new datapoint from X and seeing it would have on the splits under
    consideration, i.e., raising or lowering the impurity.
    - The confidence interval for each arm return is computed via propagation of uncertainty formulas in other fns.

    :param X: Full dataset
    :param feature_idcs: Feature indices of the dataset under consideration
    :return: Return the indices of the best feature to split on and best bin edge of that feature to split on
    """
    # Right now, we assume the number of bin edges is constant across features
    F = len(feature_idcs)  # TODO: Map back to feature idcs
    B = 11  # TODO: Fix this hard-coding
    N = len(X)
    batch_size = 100  # Right now, constant batch size
    round_count = 0

    candidates = np.array(list(itertools.product(range(F), range(B))))
    # NOTE: Instantiating these as np.inf gives runtime errors and nans.
    # TODO (@motiwari): Find a better way to do this instead of using 1000
    estimates = 1000 * np.ones((F, B))
    lcbs = 1000 * np.ones((F, B))
    ucbs = 1000 * np.ones((F, B))
    num_samples = np.zeros((F, B))
    exact_mask = np.zeros((F, B))
    cb_delta = np.zeros((F, B))

    # Create a list of histogram objects, one per feature
    histograms = []
    for f_idx in range(F):
        # Set the minimum and maximum of bins as the minimum of maximum of data of a feature
        # Can optimize by calculating min and max at the same time?
        min_bin, max_bin = np.min(X[f_idx]), np.max(X[f_idx])
        histograms.append(Histogram(feature_idcs[f_idx], num_bins=B, min_bin=min_bin, max_bin=max_bin))

    while len(candidates) > 1:
        # If we have already pulled the arms more times than the number of datapoints in the original dataset,
        # it would be the same complexity to just compute the arm return explicitly over the whole dataset.
        # Do this to avoid scenarios where it may be required to draw \Omega(N) samples to find the best arm.
        exact_accesses = np.where((num_samples + batch_size >= N) & (exact_mask == 0))
        if len(exact_accesses[0]) > 0:
            estimates[exact_accesses], _vars = sample_targets(X, exact_accesses, histograms, batch_size)
            # The confidence intervals now only contain a point, since the return has been computed exactly
            lcbs[exact_accesses] = ucbs[exact_accesses] = estimates[exact_accesses]
            exact_mask[exact_accesses] = 1
            num_samples[exact_accesses] += N

            # TODO(@motiwari): Can't use nanmin here -- why?
            cand_condition = np.where((lcbs < ucbs.min()) & (exact_mask == 0))
            candidates = np.array(list(zip(cand_condition[0], cand_condition[1])))

        if len(candidates) == 1:
            # Break here because we have found our best candidate
            break

        print(candidates)
        accesses = (candidates[:, 0], candidates[:, 1])  # Massage arm indices for use by numpy slicing
        # NOTE: cb_delta contains a value for EVERY arm, even non-candidates, so need [accesses]
        estimates[accesses], cb_delta[accesses] = sample_targets(X, accesses, histograms, batch_size)
        num_samples[accesses] += batch_size
        lcbs[accesses] = estimates[accesses] - 10 * cb_delta[accesses]
        ucbs[accesses] = estimates[accesses] + 10 * cb_delta[accesses]

        # TODO(@motiwari): Can't use nanmin here -- why?
        # BUG: Fix this since it's 2D  # TODO: Throw out nan arms!
        cand_condition = np.where((lcbs < ucbs.min()) & (exact_mask == 0))
        candidates = np.array(list(zip(cand_condition[0], cand_condition[1])))
        round_count += 1

    best_splits = zip(np.where(lcbs == np.nanmin(lcbs))[0], np.where(lcbs == np.nanmin(lcbs))[1])
    best_splits = list(best_splits)  # possible to get first elem of zip object without converting to list?
    best_split = best_splits[0]
    # Only return non-None if the best split would indeed lower impurity
    return best_split if estimates[best_split] < 0 else None


def create_data(N=1000):
    """
    Creates some toy data. The label y is randomly chosen as 0 or 1 with equal probability. The second feature is
    randomly generated with no correlation with y. The first feature is a Gaussian centered on y.

    Datasets created by this method should be split by the first feature at 0.5.

    :param N: Dataset size
    :return: dataset
    """
    y = np.random.choice([0, 1], size=(N))
    X = np.zeros((N, 3))
    X[:, 2] = y
    X[:, 0] = np.random.normal(loc=y, scale=0.2, size=N)
    X[:, 1] = np.random.rand(N)
    return X


def get_gini(zero_count: int, one_count: int, ret_var: bool = False) -> Union[Tuple[float, float], float]:
    """
    Compute the Gini impurity for a given node, where the node is represented by the number of counts of each class
    label. The Gini impurity is equal to 1 - \sum_{i=1}^k (p_i^2)

    :param zero_count: Number of zeros in the node
    :param one_count: Number of ones in the node
    :param ret_var: Whether to the variance of the estimate
    :return: the Gini impurity of the node, as well as its estimated variance if ret_var
    """
    n = zero_count + one_count
    p0 = zero_count / n
    p1 = one_count / n
    V_p0 = p0 * (1 - p0) / n  # Assuming the independence
    G = 1 - p0 ** 2 - p1 ** 2
    # This variance comes from propagation of error formula, see
    # https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Simplification
    if ret_var:
        V_G = (-2 * p0 + 2 * p1) ** 2 * V_p0
        return G, V_G
    return G


def get_entropy(zero_count: int, one_count: int, ret_var=False) -> Union[Tuple[float, float], float]:
    """
    Compute the entropy impurity for a given node, where the node is represented by the number of counts of each class
    label. The entropy impurity is equal to - \sum{i=1}^k (p_i * \log_2 p_i)

    :param zero_count: Number of zeros in the node
    :param one_count: Number of ones in the node
    :param ret_var: Whether to the variance of the estimate
    :return: the entropy impurity of the node, as well as its estimated variance if ret_var
    """
    n = zero_count + one_count
    p0 = zero_count / n
    p1 = one_count / n
    V_p0 = p0 * (1 - p0) / n
    I = - math.log(x=p0) * p0 - math.log(x=p1) * p1
    # This variance comes from propagation of error formula, see
    # https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Simplification
    if ret_var:
        V_I = (- math.log(p0) + math.log(p1)) ** 2 * V_p0
        return I, V_I
    return I


def get_variance(zero_count: int, one_count: int, ret_var=False) -> Union[Tuple[float, float], float]:
    """
    Compute the variance for a given node, where the node is represented by the number of counts of each class
    label.

    :param zero_count: Number of zeros in the node
    :param one_count: Number of ones in the node
    :param ret_var: Whether to the variance of the estimate
    :return: the variance of the node, as well as its estimated variance if ret_var
    """
    n = zero_count + one_count
    p0 = zero_count / n
    p1 = one_count / n
    V_target = p0 * (1 - p0)  # Assume that each target is from bernoulli distribution
    # This variance comes from propagation of error formula, see
    # https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Simplification
    if ret_var:
        V_V_target = (1 - 2 * p0) ** 2 * V_target
        return V_target, V_V_target
    return V_target


def get_impurity_reductions(histogram: Histogram, _bin_edge_idcs: List[int], ret_vars: bool = False,
                            impurity_measure: str = "GINI") -> Union[Tuple[float, float], float]:
    """
    Given a histogram of counts for each bin, compute the impurity reductions if we were to split a node on any of the
    histogram's bin edges.

    Impurity is measured either by Gini index or entropy

    Return impurity reduction when splitting node by bins in _bin_edge_idcs
    """

    # get_impurity is a function of measuring impurity for a node 
    if impurity_measure == "GINI":
        get_impurity: Callable = get_gini
    elif impurity_measure == "ENTROPY":
        get_impurity: Callable = get_entropy
    elif impurity_measure == "VARIANCE":
        get_impurity: Callable = get_variance
    else:
        Exception('Did not assign any measure for impurity calculation in get_impurity_reduction function')

    h = histogram
    b = len(_bin_edge_idcs)
    assert b <= h.num_bins, \
        "len(bin_edges) we want to calculate impurity reductions is greater than len(total_bin_edges"
    impurities_left = np.zeros(b)
    impurities_right = np.zeros(b)
    V_impurities_left = np.zeros(b)
    V_impurities_right = np.zeros(b)
    
    n = h.left_zeros[0] + h.left_ones[0] + h.right_zeros[0] + h.right_ones[0]
    for i in range(b):
        b_idx = _bin_edge_idcs[i]
        IL, V_IL = get_impurity(h.left_zeros[b_idx], h.left_ones[b_idx],
                                ret_var=True)
        IR, V_IR = get_impurity(h.right_zeros[b_idx], h.right_ones[b_idx]
                                , ret_var=True)
        
        # Impurity is weighted by population of each node during a split
        left_weight = (h.left_zeros[b_idx] + h.left_ones[b_idx]) / n
        right_weight = (h.right_zeros[b_idx + h.right_ones[b_idx]]) / n
        impurities_left[i], V_impurities_left[i] = left_weight * IL, left_weight ** 2 * V_IL
        impurities_right[i], V_impurities_right[i] = right_weight * IL, right_weight ** 2 * V_IR

    impurity_curr, V_impurity_curr = get_impurity(h.left_zeros[0] + h.right_zeros[0], h.left_ones[0] + h.right_ones[0]
                                                  , ret_var=True)
    # TODO(@motiwari): Might not need to subtract off impurity_curr since it doesn't affect reduction in a single feature?
    # (once best feature is determined)
    impurity_reductions = (impurities_left + impurities_right) - impurity_curr
    if ret_vars:
        # Note the last plus because Var(X-Y) = Var(X) + Var(Y) if X, Y are independent (this is an UNDERestimate)
        impurity_vars = V_impurities_left + V_impurities_right + V_impurity_curr
        return impurity_reductions, impurity_vars
    return impurity_reductions


def ground_truth_stump(X: np.ndarray, show: bool = False):
    """
    Given a dataset, perform the first step of making a tree: find the single best (feature, feature_value) pair
    to split on using the sklearn implementation.

    :param X: Dataset to build a stump out of
    :param show: Whether to display the visual plot
    :return: None
    """
    # Since y is mostly correlated with the first feature, we expect a 1-node stump to look at the first feature
    # and choose that. So if x0 < 0.5, choose 0, otherwise choose 1
    DT = DecisionTreeClassifier(max_depth=1)
    DT.fit(X[:, :2], X[:, 2])
    print(export_text(DT))
    if show:
        plot_tree(DT)
        plt.show()


def main():
    X = create_data(10000)
    ground_truth_stump(X, show=False)
    h = Histogram(0, num_bins=11)
    h.add(X)
    reductions, vars = get_impurity_reductions(h, np.arange(len(h.bin_edges)), ret_vars=True)
    print("=> THIS IS GROUND TRUTH\n")
    print(reductions)
    print(vars)
    print(np.argmin(reductions))
    # print(h[0])
    print("\n\n")

    print("=> THIS IS MAB\n")
    print(solve_mab(X, [0, 1]))


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    np.random.seed(0)
    main()

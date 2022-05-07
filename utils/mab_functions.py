import numpy as np
import itertools

from typing import List, Tuple, Callable, Union, DefaultDict
from collections import defaultdict

from data_structures.histogram import Histogram
from utils.constants import CONF_MULTIPLIER, TOLERANCE
from utils.criteria import get_gini, get_entropy, get_variance, get_mse
from utils.utils import type_check, class_to_idx, counts_of_labels, make_histograms

type_check()


def get_impurity_fn(impurity_measure: str) -> Callable:
    if impurity_measure == "GINI":
        get_impurity: Callable = get_gini
    elif impurity_measure == "ENTROPY":
        get_impurity: Callable = get_entropy
    elif impurity_measure == "VARIANCE":
        get_impurity: Callable = get_variance
    elif impurity_measure == "MSE":
        get_impurity: Callable = get_mse
    else:
        Exception(
            "Did not assign any measure for impurity calculation in get_impurity_reduction function"
        )
    return get_impurity


def get_impurity_reductions(
    is_classification: bool,
    histogram: Histogram,
    _bin_edge_idcs: List[int],
    ret_vars: bool = False,
    impurity_measure: str = "",
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Given a histogram of counts for each bin, compute the impurity reductions if we were to split a node on any of the
    histogram's bin edges.

    Impurity is measured either by Gini index or entropy

    :param is_classification: Whether the problem is a classification problem(True) or a regression problem(False)
    :returns: Impurity reduction when splitting node by bins in _bin_edge_idcs
    """
    if impurity_measure == "":
        impurity_measure = "GINI" if is_classification else "MSE"
    get_impurity = get_impurity_fn(impurity_measure)

    h = histogram
    b = len(_bin_edge_idcs)
    assert (
        b <= h.num_bins
    ), "len(bin_edges) whose impurity reductions we want to calculate is greater than len(total_bin_edges)"
    impurities_left = np.zeros(b)
    impurities_right = np.zeros(b)
    V_impurities_left = np.zeros(b)
    V_impurities_right = np.zeros(b)

    if is_classification:
        n = np.sum(h.left[0, :]) + np.sum(h.right[0, :])
    else:
        n = len(h.left_pile[0]) + len(h.right_pile[0])
    for i in range(b):
        b_idx = _bin_edge_idcs[i]
        if is_classification:
            IL, V_IL = get_impurity(h.left[b_idx, :], ret_var=True)
            IR, V_IR = get_impurity(h.right[b_idx, :], ret_var=True)
        else:
            IL, V_IL = get_impurity(h.left_pile[b_idx], ret_var=True)
            IR, V_IR = get_impurity(h.right_pile[b_idx], ret_var=True)

        # Impurity is weighted by population of each node during a split
        if is_classification:
            left_weight = np.sum(h.left[b_idx, :]) / n
            right_weight = np.sum(h.right[b_idx, :]) / n
        else:
            left_weight = len(h.left_pile[b_idx]) / n
            right_weight = len(h.right_pile[b_idx]) / n
        impurities_left[i], V_impurities_left[i] = (
            float(left_weight * IL),
            float((left_weight ** 2) * V_IL),
        )
        impurities_right[i], V_impurities_right[i] = (
            float(right_weight * IR),
            float((right_weight ** 2) * V_IR),
        )

    if is_classification:
        impurity_curr, V_impurity_curr = get_impurity(
            h.left[0, :] + h.right[0, :], ret_var=True
        )
    else:
        impurity_curr, V_impurity_curr = get_impurity(
            h.left_pile[0] + h.right_pile[0], ret_var=True
        )
    impurity_curr = float(impurity_curr)
    V_impurity_curr = float(V_impurity_curr)
    # TODO(@motiwari): Might not need to subtract off impurity_curr
    #  since it doesn't affect reduction in a single feature?
    # (once best feature is determined)
    impurity_reductions = (impurities_left + impurities_right) - impurity_curr

    if ret_vars:
        # Note the last plus because Var(X-Y) = Var(X) + Var(Y) if X, Y are independent (this is an UNDERestimate)
        impurity_vars = V_impurities_left + V_impurities_right + V_impurity_curr
        return impurity_reductions, impurity_vars
    return impurity_reductions  # Jay: we can change the type of impurity_reductions to Tuple[np.ndarray] whose each array has size 1


def sample_targets(
    is_classification: bool,
    data: np.ndarray,
    labels: np.ndarray,
    arms: Tuple[np.ndarray, np.ndarray],
    histograms: List[object],
    batch_size: int,
    impurity_measure: str = "",
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Given a dataset and set of features, draw batch_size new datapoints (with replacement) from the dataset. Insert
    their feature values into the (potentially non-empty) histograms and recompute the changes in impurity
    for each potential bin split

    :param is_classification: Whether is a classification problem(True) or regression problem(False)
    :param data: input data array with 2 dimensions
    :param labels: target data array with 1 dimension
    :param arms: arms we want to consider
    :param histograms: list of the histograms for ALL feature indices
    :param batch_size: the number of samples we're going to choose
    :param impurity_measure: A name of impurity measure
    :return: impurity_reduction and its variance of accesses
    """
    # TODO(@motiwari): Samples all bin edges for a given feature, should only sample those under consideration.
    feature_idcs, bin_edge_idcs = arms
    f2bin_dict = defaultdict(
        list
    )  # f2bin_dict[i] contains bin indices list of ith feature
    for idx in range(len(bin_edge_idcs)):
        feature = feature_idcs[idx]
        bin_edge = bin_edge_idcs[idx]
        f2bin_dict[feature].append(bin_edge)

    # NOTE: impurity_reductions and cb_deltas are smaller subsets than the original
    impurity_reductions = np.array([], dtype=float)
    cb_deltas = np.array([], dtype=float)
    N = len(data)

    sample_idcs = (
        np.arange(N) if N <= batch_size else np.random.choice(N, size=batch_size)
    )  # Default: with replacement (replace=True)
    num_queries = len(sample_idcs)  # May be less than batch_size due to truncation
    samples = data[sample_idcs]
    sample_labels = labels[sample_idcs]

    for f_idx, f in enumerate(f2bin_dict):
        h = histograms[f]
        h.add(samples, sample_labels)  # This is where the labels are used
        # TODO(@motiwari): Can make this more efficient because a lot of histogram computation is reused across steps
        i_r, cb_d = get_impurity_reductions(
            is_classification, h, f2bin_dict[f], ret_vars=True
        )
        impurity_reductions = np.concatenate([impurity_reductions, i_r])
        cb_deltas = np.concatenate(
            [cb_deltas, np.sqrt(cb_d)]
        )  # The above fn returns the vars

    # TODO(@motiwari): This seems dangerous, because access appears to be a linear index to the array
    return impurity_reductions, cb_deltas, num_queries


def verify_reduction(data: np.ndarray, labels: np.ndarray, feature, value) -> bool:
    # TODO: Fix this. Use a dictionary to store original labels -> label index
    #  or use something like label_idx,
    #  label in np.unique(labels) to avoid assuming that the labels are 0, ... K-1
    class_dict: dict = class_to_idx(np.unique(labels))
    counts: np.ndarray = counts_of_labels(
        class_dict, labels
    )  # counts[i] is the number of points that have the label class_dict[i]
    p = counts / len(labels)
    root_impurity = 1 - np.dot(p, p)

    left_idcs = np.where(data[:, feature] <= value)
    left_labels = labels[left_idcs]
    L_counts: np.ndarray = counts_of_labels(class_dict, left_labels)

    # This is already a pure node
    if len(left_idcs[0]) == 0:
        return False
    p_L = L_counts / np.sum(L_counts)

    right_idcs = np.where(data[:, feature] > value)
    right_labels = labels[right_idcs]
    R_counts: np.ndarray = counts_of_labels(class_dict, right_labels)

    # This is already a pure node
    if len(right_idcs[0]) == 0:
        return False
    p_R = R_counts / np.sum(R_counts)

    split_impurity = (1 - np.dot(p_L, p_L)) * np.sum(L_counts) + (
        1 - np.dot(p_R, p_R)
    ) * np.sum(R_counts)
    split_impurity /= len(labels)

    return TOLERANCE < root_impurity - split_impurity


def solve_mab(
    data: np.ndarray,
    labels: np.ndarray,
    discrete_bins_dict: DefaultDict,
    fixed_bin_type: str = "",
    is_classification: bool = True,
    impurity_measure: str = "",
    min_impurity_reduction: float = 0,
) -> Tuple[int, float, float, int]:
    """
    Solve a multi-armed bandit problem. The objective is to find the best feature to split on, as well as the value
    that feature should be split at.

    - The arms correspond to the (feature, feature_value) pairs. There are F x B of them.
    - The true arm return corresponds to the actual reduction in impurity if we were to perform that split.
    - Pulling an arm corresponds to drawing a new datapoint from X and seeing it would have on the splits under
    consideration, i.e., raising or lowering the impurity.
    - The confidence interval for each arm return is computed via propagation of uncertainty formulas in other fns.

    :param data: Feature set
    :param labels: Labels of datapoints
    :param discrete_bins_dict: A dictionary of discrete bins
    :param fixed_bin_type: The type of bin to use. There are 3 choices--linear, discrete, and identity.
    :param num_queries: mutable variable to update the number of datapoints queried
    :param is_classification:  Whether is a classification problem(True) or regression problem(False)
    :param impurity_measure: A name of impurity_measure
    :return: Return the indices of the best feature to split on and best bin edge of that feature to split on
    """
    F = len(data[0])
    B = 11  # TODO: Fix this hard-coding
    N = len(data)
    batch_size = 100  # Right now, constant batch size
    round_count = 0

    candidates = np.array(list(itertools.product(range(F), range(B))))
    estimates = np.empty((F, B))
    lcbs = np.empty((F, B))
    ucbs = np.empty((F, B))
    num_samples = np.zeros((F, B))
    exact_mask = np.zeros((F, B))
    cb_delta = np.zeros((F, B))

    # Make a list of histograms, a list of indices that we don't consider as potential arms, and a list of indices
    # that we consider as potential arms.
    histograms, not_considered_idcs, considered_idcs = make_histograms(
        is_classification,
        data,
        labels,
        discrete_bins_dict,
        fixed_bin_type=fixed_bin_type,
        num_bins=B,
    )

    considered_idcs = np.array(considered_idcs)
    not_considered_idcs = np.array(not_considered_idcs)
    if len(not_considered_idcs) > 0:
        not_considered_access = (not_considered_idcs[:, 0], not_considered_idcs[:, 1])
        exact_mask[not_considered_access] = 1
        lcbs[not_considered_access] = ucbs[not_considered_access] = estimates[
            not_considered_access
        ] = float("inf")
        candidates = considered_idcs

    total_queries = 0
    while len(candidates) > 1:
        # If we have already pulled the arms more times than the number of datapoints in the original dataset,
        # it would be the same complexity to just compute the arm return explicitly over the whole dataset.
        # Do this to avoid scenarios where it may be required to draw \Omega(N) samples to find the best arm.
        exact_accesses = np.where((num_samples + batch_size >= N) & (exact_mask == 0))
        if len(exact_accesses[0]) > 0:
            estimates[exact_accesses], _vars, num_queries = sample_targets(
                is_classification,
                data,
                labels,
                exact_accesses,
                histograms,
                N,
                impurity_measure=impurity_measure,
            )

            # The confidence intervals now only contain a point, since the return has been computed exactly
            lcbs[exact_accesses] = ucbs[exact_accesses] = estimates[exact_accesses]
            exact_mask[exact_accesses] = 1
            num_samples[exact_accesses] += N

            # TODO(@motiwari): Can't use nanmin here -- why?
            cand_condition = np.where((lcbs < ucbs.min()) & (exact_mask == 0))
            candidates = np.array(list(zip(cand_condition[0], cand_condition[1])))
            total_queries += num_queries

        # Last candidates were exactly computed
        if len(candidates) <= 1:
            break

        # Massage arm indices for use by numpy slicing
        accesses = (
            candidates[:, 0],
            candidates[:, 1],
        )
        # NOTE: cb_delta contains a value for EVERY arm, even non-candidates, so need [accesses]
        estimates[accesses], cb_delta[accesses], num_queries = sample_targets(
            is_classification,
            data,
            labels,
            accesses,
            histograms,
            batch_size,
            impurity_measure=impurity_measure,
        )
        num_samples[accesses] += batch_size
        lcbs[accesses] = estimates[accesses] - CONF_MULTIPLIER * cb_delta[accesses]
        ucbs[accesses] = estimates[accesses] + CONF_MULTIPLIER * cb_delta[accesses]

        # TODO(@motiwari): Can't use nanmin here -- why?
        # BUG: Fix this since it's 2D  # TODO: Throw out nan arms!
        cand_condition = np.where((lcbs < ucbs.min()) & (exact_mask == 0))
        candidates = np.array(list(zip(cand_condition[0], cand_condition[1])))
        total_queries += num_queries
        round_count += 1

    best_split = zip(
        np.where(lcbs == np.nanmin(lcbs))[0], np.where(lcbs == np.nanmin(lcbs))[1]
    ).__next__()  # Get first element
    best_feature = best_split[0]
    best_value = histograms[best_feature].bin_edges[best_split[1]]
    best_reduction = estimates[best_split]

    # Uncomment when debugging
    # if verify_reduction(
    #    data=data, labels=labels, feature=best_feature, value=best_value
    # ):
    #    return best_feature, best_value, best_reduction

    # Only return the split if it would indeed lower the impurity
    if best_reduction < min_impurity_reduction:
        return best_feature, best_value, best_reduction, total_queries
    else:
        return total_queries

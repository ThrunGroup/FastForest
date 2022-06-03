import math
import itertools
import numpy as np
import random

from typing import List, Tuple, DefaultDict
from collections import defaultdict

from utils.constants import (
    CONF_MULTIPLIER,
    TOLERANCE,
    GINI,
    MSE,
    LINEAR,
    IDENTITY,
    BATCH_SIZE,
    DEFAULT_NUM_BINS,
    RANDOM,
    MAX_SEED,
)
from utils.criteria import get_impurity_reductions
from utils.utils import (
    type_check,
    class_to_idx,
    counts_of_labels,
    make_histograms,
    empty_histograms,
)
from data_structures.histogram import Histogram

type_check()


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


def sample_targets(
    is_classification: bool,
    data: np.ndarray,
    labels: np.ndarray,
    arms: Tuple[np.ndarray, np.ndarray],
    histograms: List[object],
    batch_size: int,
    impurity_measure: str = GINI,
    population_idcs: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
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
    :param population_idcs: An array of remaining population indices. If an empty array, sample data with replacement
    :return: impurity_reduction and its variance of accesses
    """
    # TODO(@motiwari): Samples all bin edges for a given feature, should only sample those under consideration.

    feature_idcs, bin_edge_idcs = arms
    f2bin_dict = defaultdict(
        list
    )  # f2bin_dict[i] contains bin indices list of ith feature
    for idx in range(len(bin_edge_idcs)):
        # Get the corresponding feature index for this bin index in the list of (feature_idcs, bin_idcs) pairs
        feature = feature_idcs[idx]
        bin_edge = bin_edge_idcs[idx]
        f2bin_dict[feature].append(bin_edge)

    # NOTE: impurity_reductions and cb_deltas are smaller subsets than the original
    impurity_reductions = np.array([], dtype=float)
    cb_deltas = np.array([], dtype=float)
    N = len(data)

    with_replacement = population_idcs is None
    initial_pop_size = None if population_idcs is None else N
    if with_replacement:  # Sample with replacement
        if N <= batch_size:
            sample_idcs = np.arange(N, dtype=np.int64)
            initial_pop_size = N  # Since we're sampling all the samples
        else:
            sample_idcs = np.random.choice(N, size=batch_size, replace=True)
    else:
        M = len(population_idcs)
        idcs = (
            np.arange(M, dtype=np.int64)
            if batch_size >= M
            else np.random.choice(M, batch_size, replace=False)
        )
        sample_idcs = population_idcs[idcs]
        population_idcs = np.delete(
            population_idcs, idcs
        )  # Delete drawn samples from population

    samples = data[sample_idcs]
    sample_labels = labels[sample_idcs]
    num_queries = 0
    for f_idx, f in enumerate(f2bin_dict):
        h: Histogram = histograms[f]
        h.add(samples, sample_labels)  # This is where the labels are used
        num_queries += len(
            samples
        )  # num_queries should be updated per histogram insert

        # TODO(@motiwari): Can make this more efficient because a lot of histogram computation is reused across steps
        i_r, cb_d = get_impurity_reductions(
            is_classification=is_classification,
            histogram=h,
            bin_edge_idcs=f2bin_dict[f],
            ret_vars=True,
            impurity_measure=impurity_measure,
            pop_size=initial_pop_size,
        )
        impurity_reductions = np.concatenate([impurity_reductions, i_r])
        cb_deltas = np.concatenate(
            [cb_deltas, np.sqrt(cb_d)]
        )  # The above fn returns the vars

    # TODO(@motiwari): This seems dangerous, because access appears to be a linear index to the array
    return impurity_reductions, cb_deltas, num_queries, population_idcs


def solve_exactly(
    data: np.ndarray,
    labels: np.ndarray,
    minmax: Tuple[np.ndarray, np.ndarray] = None,
    discrete_bins_dict: DefaultDict = None,
    binning_type: str = IDENTITY,
    num_bins: int = DEFAULT_NUM_BINS,
    is_classification: bool = True,
    impurity_measure: str = GINI,
    min_impurity_reduction: float = 0,
) -> Tuple[int, float, float, int]:
    """
    Find the best feature to split on, as well as the value that feature should be split at, using the canonical (exact)
    algorithm. We assume that we still discretize (which is not true in RF).

    This method may be inefficient because we rely on the same sample_targets that the MAB solution uses. Nonetheless,
    it gives an accurate measure of the total number of queries and is used for ease of implementation.

    :param data: Feature set
    :param labels: Labels of datapoints
    :param minmax: (minimum array of features, maximum array of features)
    :param discrete_bins_dict: A dictionary of discrete bins
    :param binning_type: The type of bin to use. There are 3 choices--linear, discrete, and identity.
    :param num_queries: mutable variable to update the number of datapoints queried
    :param is_classification:  Whether is a classification problem(True) or regression problem(False)
    :param impurity_measure: A name of impurity_measure
    :param impurity_measure: Minimum impurity reduction beyond which to return a nonempty solution
    :return: Return the indices of the best feature to split on and best bin edge of that feature to split on
    """
    N = len(data)
    F = len(data[0])

    if binning_type == IDENTITY:
        B = len(data)
    elif binning_type == RANDOM:
        assert (
            num_bins is None
        ), "When using Extremely Random Forests, please pass num_bins=None explicitly. If you want to set a custom \
                            number of Extremely Random bins, please update callsites and remove this assertion."
        if is_classification:
            B = math.ceil(np.sqrt(F))
        else:
            # TODO(@motiwari): change this back to F after ddl
            # B = math.ceil(np.sqrt(F))
            B = F
    else:
        B = num_bins

    candidates = np.array(list(itertools.product(range(F), range(B))))
    estimates = np.empty((F, B))

    # Make a list of histograms, a list of indices that we don't consider as potential arms, and a list of indices
    # that we consider as potential arms.
    histograms, not_considered_idcs, considered_idcs = make_histograms(
        is_classification=is_classification,
        data=data,
        labels=labels,
        minmax=minmax,
        discrete_bins_dict=discrete_bins_dict,
        binning_type=binning_type,
        num_bins=B,
    )

    considered_idcs = np.array(considered_idcs)
    not_considered_idcs = np.array(not_considered_idcs)
    if len(not_considered_idcs) > 0:
        not_considered_access = (not_considered_idcs[:, 0], not_considered_idcs[:, 1])
        estimates[not_considered_access] = float("inf")
        candidates = considered_idcs

    # Massage arm indices for use by numpy slicing
    accesses = (
        candidates[:, 0],
        candidates[:, 1],
    )

    estimates[accesses], _cb_delta, num_queries, _ = sample_targets(
        is_classification=is_classification,
        data=data,
        labels=labels,
        arms=accesses,
        histograms=histograms,
        batch_size=N,
        impurity_measure=impurity_measure,
    )

    total_queries = num_queries
    # TODO(@motiwari): Can't use nanmin here -- why?
    # BUG: Fix this since it's 2D  # TODO: Throw out nan arms!
    best_split = zip(
        np.where(estimates == np.nanmin(estimates))[0],
        np.where(estimates == np.nanmin(estimates))[1],
    ).__next__()  # Get first element
    best_feature = best_split[0]
    best_value = histograms[best_feature].bin_edges[best_split[1]]
    best_reduction = estimates[best_split]

    if best_reduction < min_impurity_reduction:
        return best_feature, best_value, best_reduction, total_queries
    else:
        return total_queries


def solve_mab(
    data: np.ndarray,
    labels: np.ndarray,
    minmax: Tuple[np.ndarray, np.ndarray] = None,
    discrete_bins_dict: DefaultDict = None,
    binning_type: str = LINEAR,
    num_bins: int = DEFAULT_NUM_BINS,
    is_classification: bool = True,
    impurity_measure: str = GINI,
    min_impurity_reduction: float = 0,
    epsilon=0.01,
    with_replacement: bool = False,
    budget: int = None,
    verbose: bool = False,
    batch_size: int = BATCH_SIZE,
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
    :param minmax: (minimum array of features, maximum array of features).
    :param discrete_bins_dict: A dictionary of discrete bins
    :param binning_type: The type of bin to use. There are 3 choices--linear, discrete, and identity.
    :param num_queries: mutable variable to update the number of datapoints queried
    :param is_classification:  Whether is a classification problem(True) or regression problem(False)
    :param impurity_measure: A name of impurity_measure
    :param with_replacement: Whether to sample with replacement.
    :return: Return the indices of the best feature to split on and best bin edge of that feature to split on
    """
    N = len(data)
    F = len(data[0])
    if binning_type == IDENTITY:
        raise Exception(
            "You're running solve_mab without histogramming. Did you mean to?"
        )
        B = N
    elif binning_type == RANDOM:
        assert (
            num_bins is None
        ), "When using Extremely Random Forests, please pass num_bins=None explicitly. If you want to set a custom \
                    number of Extremely Random bins, please update callsites and remove this assertion."
        if is_classification:
            B = math.ceil(np.sqrt(F))
        else:
            # TODO(@motiwari): change this back to F after ddl
            # B = math.ceil(np.sqrt(F))
            # B = int(F / 3)
            B = F
    else:
        B = num_bins

    batch_size = batch_size
    round_count = 0
    population_idcs = None if with_replacement is True else np.arange(N, dtype=np.int64)
    if impurity_measure == "":
        impurity_measure = GINI if is_classification else MSE

    if verbose:
        print("Number of arms:", B * F)
    candidates = np.array(list(itertools.product(range(F), range(B))))
    estimates = np.empty((F, B))
    lcbs = np.empty((F, B))
    ucbs = np.empty((F, B))
    num_samples = np.zeros((F, B), dtype=np.int64)
    exact_mask = np.zeros((F, B))
    cb_delta = np.zeros((F, B))

    # Make a list of histograms, a list of indices that we don't consider as potential arms, and a list of indices
    # that we consider as potential arms.
    histograms, not_considered_idcs, considered_idcs = make_histograms(
        is_classification=is_classification,
        data=data,
        labels=labels,
        minmax=minmax,
        discrete_bins_dict=discrete_bins_dict,
        binning_type=binning_type,
        num_bins=B,
    )

    considered_idcs = np.array(considered_idcs)
    not_considered_idcs = np.array(not_considered_idcs)
    if len(not_considered_idcs) > 0:
        not_considered_access = (not_considered_idcs[:, 0], not_considered_idcs[:, 1])
        exact_mask[not_considered_access] = 1
        ucbs[not_considered_access] = estimates[not_considered_access] = float("inf")
        lcbs[not_considered_access] = -float("inf")
        candidates = considered_idcs

    total_queries = 0
    while len(candidates) > 5:
        # If we have already pulled the arms more times than the number of datapoints in the original dataset,
        # it would be the same complexity to just compute the arm return explicitly over the whole dataset.
        # Do this to avoid scenarios where it may be required to draw \Omega(N) samples to find the best arm.
        if with_replacement:
            # raise Exception("Did you really want to sample with replacement?")
            exact_accesses = np.where(
                (num_samples + batch_size >= N) & (exact_mask == 0)
            )
            if len(exact_accesses[0]) > 0:
                empty_histograms(histograms, exact_accesses)
                estimates[exact_accesses], _vars, num_queries, _ = sample_targets(
                    is_classification=is_classification,
                    data=data,
                    labels=labels,
                    arms=exact_accesses,
                    histograms=histograms,
                    batch_size=N,
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

                # check that we're still within budget
                if (budget is not None) and (budget - total_queries <= 0):
                    break

                # None of the CIs overlap with 0. We are confident that there is no possible impurity reduction.
                if lcbs.min() > 0:
                    break

        # Last candidates were exactly computed
        if len(candidates) <= 1:
            break
        if population_idcs is not None and len(population_idcs) == 0:
            lcbs = ucbs = estimates
            break

        # Massage arm indices for use by numpy slicing
        accesses = (
            candidates[:, 0],
            candidates[:, 1],
        )
        # NOTE: cb_delta contains a value for EVERY arm, even non-candidates, so need [accesses]

        (
            estimates[accesses],
            cb_delta[accesses],
            num_queries,
            population_idcs,
        ) = sample_targets(
            is_classification=is_classification,
            data=data,
            labels=labels,
            arms=accesses,
            histograms=histograms,
            batch_size=batch_size,
            impurity_measure=impurity_measure,
            population_idcs=population_idcs,
        )
        num_samples[accesses] += batch_size
        total_queries += num_queries
        lcbs[accesses] = estimates[accesses] - CONF_MULTIPLIER * cb_delta[accesses]
        ucbs[accesses] = estimates[accesses] + CONF_MULTIPLIER * cb_delta[accesses]

        # check that we're still within budget
        if (budget is not None) and (budget - total_queries <= 0):
            break

        # None of the CIs overlap with 0. We are confident that there is no possible impurity reduction.
        if lcbs.min() > 0:
            break

        # TODO(@motiwari): Can't use nanmin here -- why?
        # BUG: Fix this since it's 2D  # TODO: Throw out nan arms!
        tied_arms = np.zeros((F, B))
        tied_arms[np.where((lcbs < (1 - epsilon) * estimates.min()))] = 1
        cand_condition = np.where(
            (exact_mask == 0) & (lcbs < ucbs.min()) & (lcbs < min_impurity_reduction) & (tied_arms == 0)
        )
        candidates = np.array(list(zip(cand_condition[0], cand_condition[1])))
        round_count += 1

    best_splits = zip(
        np.where(estimates == np.nanmin(estimates))[0],
        np.where(estimates == np.nanmin(estimates))[1],
    )
    best_split = random.choice(list(best_splits))
    best_feature = best_split[0]
    best_value = histograms[best_feature].bin_edges[best_split[1]]
    best_reduction = estimates[best_split]
    if verbose:
        print("Best reduction:", best_reduction)
        print("Round count:", round_count)

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


def filter_tied_arms(candidates: np.ndarray, tied_arms, F, B):
    """
    Removed the tied_arms from the candidates. Assumes first index corresponds to feature and second corresponds to bin.

    :param candidates:
    :param tied_arms:
    :return:
    """
    cand_flattened_indices = [
        B * candidate[0] + candidate[1] for candidate in candidates
    ]
    tied_flattened_indices = [B * tied[0] + tied[1] for tied in tied_arms]

    filtered_cand_indices = np.setdiff1d(cand_flattened_indices, tied_flattened_indices)
    filtered_candidates = np.array(
        [[f_c_ind // B, f_c_ind % B] for f_c_ind in filtered_cand_indices]
    )
    return filtered_candidates

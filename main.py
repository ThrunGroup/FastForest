import sys
import numpy as np
import itertools
import math

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text


# TODO: Define histogram class
def sample_targets(X, feature_idcs, histograms, batch_size):
    """
    Given a dataset and set of features, draw batch_size new datapoints (with replacement) from the dataset. Insert
    their feature values into the (potentially non-empty) histograms and recompute the changes in impurity
    for each potential bin split
    """

    # TODO(@motiwari): Samples all bin edges for a given feature, should only sample those under consideration.

    # histograms variable should be a list of histograms, one per feature
    assert len(histograms) == len(feature_idcs), "Need one histogram per feature"
    F = len(feature_idcs)  # TODO: Map back to feature idcs
    B = 11  # TODO: Fix this hardcoding
    N = len(X)
    sample_idcs = np.random.choice(N, size=batch_size)  # Default: with replacement (replace=True)
    samples = X[sample_idcs]
    impurity_reductions = np.zeros((F, B))
    cb_deltas = np.zeros((F, B))
    for f_idx, f in enumerate(feature_idcs):
        h = histograms[f_idx]
        h = add_to_histogram(samples, f, h)  # This is where the labels are used
        # TODO(@motiwari): Can make this more efficient because a lot of histogram computation is reused across steps
        impurity_reductions[f_idx, :], cb_deltas[f_idx, :] = get_impurity_reductions(h, ret_vars=True)
        cb_deltas[f_idx, :] = np.sqrt(cb_deltas[f_idx, :])  # The above fn returns the vars

    # TODO(@motiwari): This seems dangerous, because access appears to be a linear index to the array
    return impurity_reductions.reshape(-1), cb_deltas.reshape(-1)


def solve_mab(X, feature_idcs):
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
    B = 11  # TODO: Fix this hardcoding
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

    # Create a list of histograms, one per feature
    histograms = []
    for f_idx in range(F):
        histograms.append(create_histogram(X, feature_idcs[f_idx], middle_bins=10))

    while len(candidates) > 0:
        # If we have already pulled the arms more times than the number of datapoints in the original dataset,
        # it would be the same complexity to just compute the arm return explicitly over the whole dataset.
        # Do this to avoid scenarios where it may be required to draw \Omega(N) samples to find the best arm.
        comp_exactly_condition = np.where((num_samples + batch_size >= N) & (exact_mask == 0))
        compute_exactly = np.array(list(zip(comp_exactly_condition[0], comp_exactly_condition[1])))
        if len(compute_exactly) > 0:
            exact_accesses = (compute_exactly[:, 0], compute_exactly[:, 1])
            estimates[exact_accesses], _vars = sample_targets(X, feature_idcs, histograms, batch_size)
            # The confidence intervals now only contain a point, since the return has been computed exactly
            lcbs[exact_accesses] = ucbs[exact_accesses] = estimates[exact_accesses]
            exact_mask[exact_accesses] = 1
            num_samples[exact_accesses] += N

            # TODO(@motiwari): Can't use nanmin here -- why?
            cand_condition = np.where((lcbs < ucbs.min()) & (exact_mask == 0))
            candidates = np.array(list(zip(cand_condition[0], cand_condition[1])))

        if len(candidates) == 0:
            # Break here because the last candidates were computed exactly
            break

        accesses = (candidates[:, 0], candidates[:, 1])  # Massage arm indices for use by numpy slicing
        # NOTE: cb_delta contains a value for EVERY arm, even non-candidates, so need [accesses]
        estimates[accesses], cb_delta[accesses] = sample_targets(X, feature_idcs, histograms, batch_size)
        num_samples[accesses] += batch_size
        lcbs[accesses] = estimates[accesses] - cb_delta[accesses]
        ucbs[accesses] = estimates[accesses] + cb_delta[accesses]

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


def get_gini(zero_count, one_count, ret_var=False):
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
    V_p0 = p0 * (1 - p0) / n
    G = 1 - p0 ** 2 - p1 ** 2
    # This variance comes from propagation of error formula, see
    # https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Simplification
    # This makes a number of assumptions which are likely unreasonable, like independence, so this estimate is likely an
    # UNDERestimate
    if ret_var:
        V_G = (-2 * p0 -2 * p1) ** 2 * V_p0
        return G, V_G
    return G


def get_entropy(zero_count, one_count, ret_var=False):
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
    # This makes a number of assumptions which are likely unreasonable, like independence, so this estimate is likely an
    # UNDERestimate
    if ret_var:
        V_I = (- math.log(p0) - 1 + math.log(p1) + 1) ** 2 * V_p0
        return I, V_I
    return I


def get_variance(zero_count, one_count, ret_var=False):
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
    V_p0 = p0 * (1 - p0) / n
    V = V_p0 + V_p1  # Assuming the independence
    # This variance comes from propagation of error formula, see
    # https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Simplification
    # This makes a number of assumptions which are likely unreasonable, like independence, so this estimate is likely an
    # UNDERestimate
    if ret_var:
        V_V = 4 / n**2 * (1 - 2*p0) ^ 2 * V_p0
        return V, V_V
    return V


def get_impurity_reductions(histogram, ret_vars=False, impurity_measure="GINI"):
    """
    Given a histogram of counts for each bin, compute the impurity reductions if were to split a node on any of the
    histogram's bin edges.

    Walking from the leftmost bin to the rightmost bin, this function keeps a running tally of the number of zeros and
    ones to the left and right of each bin edge. The sum of impurities from the two nodes that would result in
    splitting at each bin edge is calculated and compared to the original node's impurity.

    Impurity is measured either by Gini index or entropy
    """

    # get_impurity is a function of measuring impurity for a node 
    if impurity_measure == "GINI":
        get_impurity = get_gini
    elif impurity_measure == "ENTROPY":
        get_impurity = get_entropy
    elif impurity_measure == "VARIANCE":
        get_impurity = get_variance
    else:
        Exception('Did not assign any measure for impurity calculation in get_impurity_reduction function')

    bin_edges, zeros, ones = histogram
    assert len(zeros) == len(ones) == len(bin_edges) + 1, "Histogram is malformed"
    B = len(bin_edges)  # If there are 10 middle_bins, there will be 11 binedges and 12 total bins
    impurities_left = np.zeros(B)  # We can split at any bin edge (11)
    V_impurities_left = np.zeros(B)
    impurities_right = np.zeros(B)  # We can split at any bin edge (11)
    V_impurities_right = np.zeros(B)

    L0 = 0
    L1 = 0
    L_n = 0
    R0 = np.sum(zeros)
    R1 = np.sum(ones)
    R_n = np.sum(zeros) + np.sum(ones)

    impurity_curr, V_impurity_curr = get_impurity(R0, R1, ret_var=True)
    # Walk from leftmost bin to rightmost
    for b_idx in range(B):
        L0 += zeros[b_idx]
        L1 += ones[b_idx]
        L_n += zeros[b_idx] + ones[b_idx]
        R0 -= zeros[b_idx]
        R1 -= ones[b_idx]
        R_n -= zeros[b_idx] + ones[b_idx]

        I_L, V_I_L = get_impurity(L0, L1, ret_var=True)
        impurities_left[b_idx], V_impurities_left[b_idx] = I_L, V_I_L
        I_R, V_I_R = get_impurity(R0, R1, ret_var=True)
        impurities_right[b_idx], V_impurities_right[b_idx] = I_R, V_I_R

    # TODO(@motiwari): Might not need to subtract off impurity_curr since it doesn't affect reduction in a single feature?
    # (once best feature is determined)
    impurity_reductions = (impurities_left + impurities_right) - impurity_curr
    if ret_vars:
        # Note the last plus because Var(X-Y) = Var(X) + Var(Y) if X, Y are independent (this is an UNDERestimate)
        impurity_vars = V_impurities_left + V_impurities_right + V_impurity_curr
        return impurity_reductions, impurity_vars
    return impurity_reductions


# TODO: Make sure feature_idex is consistent, like in histogram class for idx
def add_to_histogram(X, feature_idx, histogram):
    """
    Given the full dataset and feature index, as well as the existing histogram for that feature, add the all the
    points in the dataset to the histogram.

    Right now, it walks through each bin to find the correct one; this should be changed to a binary search.

    :param X: Full dataset to be histogrammed
    :param feature_idx: Index of the feature to analze
    :param histogram: Existing histogram for the feature (should allow None values)
    :return: None, but modify the histogram to include the relevant feature values
    """
    feature_values = X[:, feature_idx]
    Y = X[:, -1]
    bin_edges, zeros, ones = histogram
    assert len(zeros) == len(ones) == len(bin_edges) + 1, "Histogram is malformed"

    for idx, f in enumerate(feature_values):
        y = Y[idx]
        count_bucket = zeros if y == 0 else ones
        assigned = False
        # TODO(@motiwari): Change this to a binary search
        # TODO: Edge cases with feature value equal to leftmost or rightmost edges
        for b_e_idx in range(len(bin_edges)):
            if b_e_idx < len(bin_edges):
                b_e = bin_edges[b_e_idx]
                if f < b_e:  # Using < instead of <= prefers the right bucket. Revisit this line; should it be <= ?
                    count_bucket[b_e_idx] += 1
                    assigned = True
                    break

        # If the value wasn't less than any bin edge, it's greater than the max bin edge and put it in the last bin
        if not assigned:
            count_bucket[-1] += 1

    return bin_edges, zeros, ones


def create_histogram(X, feature_idx, middle_bins=10):
    """
    Create an empty histogram out of the given dataset and features with the given number of bins.

    :param X: Full dataset to be histogrammed
    :param feature_idx: Index of the feature to analze
    :param bins: Number of dividers. Note that there will actually be (bins + 2) bins in the histogram because we need
    to include each tail. E.g., if middle_bins=2 is passed and the edges are 0, 1, and 2, there will be four bins: <0, 0 to 1,
    1 to 2, and >2.
    """
    bin_edges, zeros, ones = create_empty_histogram(X, feature_idx, middle_bins=10)

    # TODO(@motiwari): Remove this to a separate call. Create_histogram should only create blank ones.
    bin_edges, zeros, ones = add_to_histogram(X, feature_idx, (bin_edges, zeros, ones))
    return bin_edges, zeros, ones


def create_empty_histogram(X, feature_idx, middle_bins=10):
    """
    Create an empty (unfilled) histogram from the given dataset and feature index.

    :param X: Full dataset
    :param feature_idx: Index of the feature for which to create a histogram
    :param middle_bins: Number of bins in the middle, excluding tails. There will be middle_bins + 1 bin edges and
    middle_bins + 2 total bins, one for each tail.
    :return:
    """
    # TODO: Don't hardcode these edges, maybe use max and min feature values?
    bin_edges = np.linspace(0.0, 1.0, middle_bins + 1)  # this creates middle_bins + 2 virtual bins to include tails

    # zeros should contain the number of zeros to the left of every bin edge, except for the last element which contains
    # the number of zeros to the right of the max. Similarly for ones.
    zeros = np.zeros(middle_bins + 2, dtype=np.int32)  # + 2 for tails
    ones = np.zeros(middle_bins + 2, dtype=np.int32)  # + 2 for tails
    return bin_edges, zeros, ones


def ground_truth_stump(X, show=False):
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
    h = create_histogram(X, 0)
    reductions, vars = get_impurity_reductions(h, ret_vars=True)
    print(reductions)
    print(vars)
    print(np.argmin(reductions))
    print(h[0])
    print(solve_mab(X, [0, 1]))


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    np.random.seed(0)
    main()

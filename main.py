import sys
import numpy as np
import itertools

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# TODO: Define histogram class


def sample_targets(X, feature_idcs, histograms, batch_size):
    """
    - Takes in a set of arms (accesses) ---> broadcast across all bins though
    - Takes existing histograms
    - Takes batch_size of points to insert into histograms
    - Inserts the points into the histograms
    - Returns estimates and confidence intervals using binomial confidence interval and propagation of error
    """
    # histograms should be a list of histograms
    assert len(histograms) == len(feature_idcs), "Need one histogram per feature"
    F = len(feature_idcs)  # TODO: Map back to feature idcs
    B = 12  # TODO: Fix this hardcoding
    samples = np.random.choice(X, size=batch_size)
    gini_reductions = np.zeros(F, B)
    cb_deltas = np.zeros(F, B)

    for f_idx, f in enumerate(feature_idcs):
        h = histograms[f_idx]
        h = add_to_histogram(samples, f, h)
        gini_reductions[f_idx, :], cb_deltas[f_idx, :] = get_gini_reductions(h, ret_confidences=True)  # TODO(@motiwari): Can make this more efficient because a lot of histogram computation is reused across steps
    return

def solve_MAB(X, feature_idcs, bin_edges):
    # Right now, we assume the number of bin edges is constant across features
    F = len(feature_idcs)  # TODO: Map back to feature idcs
    N = len(X)
    B = 12  # TODO: Fix this hardcoding
    assert bin_edges.shape == (F, B)
    batch_size = 100  # Right now, constant batch size
    p = 0
    round_count = 0

    candidates = np.array(list(itertools.product(range(k), range(N))))
    # NOTE: Instantiating these as np.inf gives runtime errors and nans. Find a better way to do this instead of using 1000
    estimates = 1000 * np.ones((F, B))
    lcbs = 1000 * np.ones((F, B))
    ucbs = 1000 * np.ones((F, B))
    T_samples = np.zeros((F, B))
    exact_mask = np.zeros((F, B))

    histograms = []
    for f_idx in range(F):
        histograms.append(create_histogram(X, feature_idcs[f], bins=10))

    while len(candidates) > 0:
        comp_exactly_condition = np.where((T_samples + batch_size >= N) & (exact_mask == 0))
        compute_exactly = np.array(list(zip(comp_exactly_condition[0], comp_exactly_condition[1])))
        if len(compute_exactly) > 0:
            exact_accesses = (compute_exactly[:, 0], compute_exactly[:, 1])
            estimates[exact_accesses], _, histograms = sample_targets(X, feature_idcs, bin_edges, histograms, batch_size)
            lcbs[exact_accesses] = estimates[exact_accesses]
            ucbs[exact_accesses] = estimates[exact_accesses]
            exact_mask[exact_accesses] = 1
            T_samples[exact_accesses] += N

            cand_condition = np.where((lcbs < ucbs.min()) & (exact_mask == 0))
            candidates = np.array(list(zip(cand_condition[0], cand_condition[1])))

        # The last candidates were computed exactly
        if len(candidates) == 0:
            break

        accesses = (candidates[:, 0], candidates[:, 1])
        new_samples, sigmas, histograms = sample_targets(X, feature_idcs, bin_edges, histograms, batch_size)
        sigmas = sigmas.reshape(F, B)  # So that can access it with sigmas[accesses] below

        # Update running average of estimates and confidence bounce
        estimates[accesses] = ((T_samples[accesses] * estimates[accesses]) + (batch_size * new_samples)) / (batch_size + T_samples[accesses])
        T_samples[accesses] += batch_size
        # NOTE: Sigmas contains a value for EVERY arm, even non-candidates, so need [accesses]
        cb_delta = sigmas[accesses] * np.sqrt(np.log(1 / p) / T_samples[accesses])
        lcbs[accesses] = estimates[accesses] - cb_delta
        ucbs[accesses] = estimates[accesses] + cb_delta

        cand_condition = np.where((lcbs < ucbs.min()) & (exact_mask == 0))  # BUG: Fix this since it's 2D
        candidates = np.array(list(zip(cand_condition[0], cand_condition[1])))
        round_count += 1

    # Choose the minimum amongst all losses and perform the swap
    # NOTE: possible to get first elem of zip object without converting to list?
    best_splits = zip(np.where(lcbs == lcbs.min())[0], np.where(lcbs == lcbs.min())[1])
    best_splits = list(best_splits)
    best_split = best_splits[0]
    return best_split


def create_data(N=1000):
    y = np.random.choice([0, 1], size=(N))
    X = np.zeros((N, 3))
    X[:, 2] = y
    X[:, 0] = np.random.normal(loc=y, scale=0.2, size=N)
    X[:, 1] = np.random.rand(N)
    return X

def get_gini(zero_count, one_count, ret_var=False):
    n = zero_count + one_count
    p0 = zero_count / n
    p1 = one_count / n
    V_p0 = p0 * (1 - p0) / n
    V_p1 = p1 * (1 - p1) / n
    G = 1 - p0**2 - p1**2
    V_G = 4*(p0**2)*V_p0 + 4*(p1**2)*V_p1
    if ret_var:
        return G, V_G
    return G

def get_gini_reductions(histogram, ret_vars=False):
    bin_edges, zeros, ones = histogram
    assert len(zeros) == len(ones) == len(bin_edges) + 1, "Histogram is malformed"
    B = len(bin_edges)  # 11
    ginis_left = np.zeros(B)  # 11
    V_ginis_left = np.zeros(B)
    ginis_right = np.zeros(B)  # 11
    V_ginis_right = np.zeros(B)
    cb_deltas = np.zeros(B)  # TODO(@motiwari): Only do this if ret_confidences

    L0 = 0
    L1 = 0
    L_n = 0
    R0 = np.sum(zeros)
    R1 = np.sum(ones)
    R_n = np.sum(zeros) + np.sum(ones)

    gini_curr, V_gini_curr = get_gini(R0, R1, ret_var=True)

    # Walk from left to right
    for b_idx in range(B):
        L0 += zeros[b_idx]
        L1 += ones[b_idx]
        L_n += zeros[b_idx] + ones[b_idx]
        R0 -= zeros[b_idx]
        R1 -= ones[b_idx]
        R_n -= zeros[b_idx] + ones[b_idx]

        G_L, V_G_L = get_gini(L0, L1, ret_var=True)
        ginis_left[b_idx], V_ginis_left[b_idx] = G_L, V_G_L
        G_R, V_G_R = get_gini(R0, R1, ret_var=True)
        ginis_right[b_idx], V_ginis_right[b_idx] = G_R, V_G_R

    gini_reductions = (ginis_left + ginis_right) - gini_curr
    gini_vars = V_ginis_left + V_ginis_right + V_gini_curr  # Note the last plus

    if ret_vars:
        return gini_reductions, gini_vars
    return gini_reductions
    # best_split_idx = np.argmin(gini_reductions)
    # if gini_reductions[best_split_idx] > 0:  # We should not split the node because any way of doing so would increase impurity, e.g., best_split is B + 1
    #     return None
    # else:
    #     return bin_edges[best_split_idx], best_split_idx


def add_to_histogram(X, feature_idx, histogram):  # TODO: Make sure feature_idex is consistent, like in histogram class for idx
    feature_values = X[:, feature_idx]
    Y = X[:, -1]

    bin_edges, zeros, ones = histogram
    assert len(zeros) == len(ones) == len(bin_edges) + 1, "Histogram is malformed"

    for idx, f in enumerate(feature_values):
        # TODO(@motiwari): Change this to a binary search
        y = Y[idx]
        count_bucket = zeros if y == 0 else ones
        assigned = False
        for b_e_idx in range(len(bin_edges)):
            if b_e_idx < len(bin_edges):
                b_e = bin_edges[b_e_idx]
                if f < b_e:  # Using < instead of <= prefers the right bucket # Causes problems!! Use <=!!!
                    count_bucket[b_e_idx] += 1
                    assigned = True
                    break
        if not assigned:
            count_bucket[-1] += 1

    return bin_edges, zeros, ones


def create_histogram(X, feature_idx, bins=10):
    """
    TODO: Edge cases with feature value equal to leftmost or rightmost edges
    """
    feature_values = X[:, feature_idx]
    Y = X[:, -1]

    max = feature_values.max()
    min = feature_values.min()

    # TODO: Don't hardcode these edges
    bin_edges = np.linspace(0.0, 1.0, bins + 1)  # this actually creates bin_edges + 2 virtual bins, for tails too
    zeros = np.zeros(bins + 2, dtype=np.int32)  # + 2 for tails
    ones = np.zeros(bins + 2, dtype=np.int32)  # + 2 for tails

    # Zeros should contain the number of zeros to the LEFT of every bin edge, except for last element which counts to the right of max
    # Ones should contain the number of ones to the LEFT of every bin edge, except for last element which counts to the right of max
    bin_edges, zeros, ones = add_to_histogram(X, feature_idx, (bin_edges, zeros, ones))
    return bin_edges, zeros, ones

def create_empty_histogram():
    raise NotImplementedError("Not yet!")
    

def ground_truth(X, show=False):
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
    ground_truth(X, show=False)
    h = create_histogram(X, 0)
    reductions, vars = get_gini_reductions(h, ret_vars=True)
    print(reductions)
    print(vars)
    print(np.argmin(reductions))
    print(h[0])


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    np.random.seed(0)
    main()
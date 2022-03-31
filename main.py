import sys
import numpy as np

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# TODO: Define histogram class


def create_data(N=1000):
    y = np.random.choice([0, 1], size=(N))
    x1 = np.random.rand(N)  # whether x1 disagrees or not

    # TODO: Optimize
    # TODO: Add floor and ceiling
    for i in range(len(x1)):
        x1[i] = y[i] if x1[i] < 0.8 else 1 - y[i]
    x1 += np.random.normal(loc=0, scale=0.1, size=(N))  # Add noise
    x1 = np.minimum(1, np.maximum(x1, 0))  # Different from np.min and np.max!

    # TODO: Optimize
    # TODO: Add floor and ceiling
    x2 = np.random.rand(N)
    for i in range(len(x2)):
        x2[i] = y[i] if x2[i] < 0.5 else 1 - y[i]
    x2 += np.random.normal(loc=0, scale=0.1, size=(N))  # Add noise
    x2 = np.minimum(1, np.maximum(x2, 0))  # Different from np.min and np.max!

    dataset = np.zeros((N, 3))
    dataset[:, 0] = x1
    dataset[:, 1] = x2
    dataset[:, 2] = y

    # print(len(np.where(dataset[:, 1] != dataset[:, 2])[0])) # Sanity check
    # print(len(np.where(dataset[:, 0] != dataset[:, 2])[0])) # Sanity check
    return dataset

def get_best_split(histogram):
    bin_edges, zeros, ones = histogram
    assert len(zeros) == len(ones) == len(bin_edges) + 1, "Histogram is malformed"
    B = len(bin_edges)  # 11
    ginis_left = np.zeros(B)  # 11
    ginis_right = np.zeros(B)  # 11

    L0 = 0
    L1 = 0
    L_n = 0
    R0 = np.sum(zeros)
    R1 = np.sum(ones)
    R_n = np.sum(zeros) + np.sum(ones)

    curr_gini = 1 - (R0/R_n)**2 - (R1/R_n)**2 # NOTE: Using R's here because they count the whole node right now

    # Walk from left to right
    for b_idx in range(B):
        L0 += zeros[b_idx]
        L1 += ones[b_idx]
        L_n += zeros[b_idx] + ones[b_idx]
        R0 -= zeros[b_idx]
        R1 -= ones[b_idx]
        R_n -= zeros[b_idx] + ones[b_idx]
        import ipdb; ipdb.set_trace()

        ginis_left[b_idx] = 1 - (L0/L_n)**2 - (L1/L_n)**2
        ginis_right[b_idx] = 1 - (R0/R_n)**2 - (R1/R_n)**2

    gini_reductions = (ginis_left + ginis_right) - curr_gini
    print(gini_reductions)
    best_split_idx = np.argmax(gini_reductions)
    if gini_reductions[best_split_idx] > 0:  # We should not split the node because any way of doing so would increase impurity, e.g., best_split is B + 1
        return None
    else:
        return bin_edges[best_split_idx], best_split_idx


def create_histogram(X, feature_idx, bins=10):
    """
    TODO: Edge cases with feature value equal to leftmost or rightmost edges
    """
    feature = X[:, feature_idx]
    Y = X[:, -1]

    max = feature.max()
    min = feature.min()
    bin_edges = np.linspace(min, max, bins + 1)  # this actually creates bin_edges + 2 virtual bins, for tails too
    zeros = np.zeros(bins + 2, dtype=np.int32)  # + 2 for tails
    ones = np.zeros(bins + 2, dtype=np.int32)  # + 2 for tails

    # Zeros should contain the number of zeros to the LEFT of every bin edge, except for last element which counts to the right of max
    # Ones should contain the number of ones to the LEFT of every bin edge, except for last element which counts to the right of max
    for idx, f in enumerate(feature):
        # TODO(@motiwari): Change this to a binary search
        y = Y[idx]
        count_bucket = zeros if y == 0 else ones
        assigned = False
        for b_e_idx in range(len(bin_edges) + 1):
            if b_e_idx < len(bin_edges):
                b_e = bin_edges[b_e_idx]
                if f < b_e:  # Using < instead of <= prefers the right bucket # Causes problems!! Use <=!!!
                    count_bucket[b_e_idx] += 1
                    assigned = True
                    break
            else:
                assert b_e_idx == len(bin_edges)
                if f <= b_e:
                    count_bucket[b_e_idx - 1] += 1 # Truncate the last bin
                    assigned = True
        if not assigned:
            raise Exception("error")
    print(bin_edges)
    print(zeros)
    print(ones)
    return bin_edges, zeros, ones
    

def best_arm(X, features, batch_size):
    pass

def update_histogram():
    pass

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
    print(get_best_split(create_histogram(X, 0)))


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    np.random.seed(0)
    main()
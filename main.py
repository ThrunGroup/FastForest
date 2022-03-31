import sys
import numpy as np

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def create_data():
    y = np.random.choice([0, 1], size=(1000))
    x1 = np.random.rand(1000)  # whether x1 disagrees or not

    # TODO: Optimize
    for i in range(len(x1)):
        x1[i] = y[i] if x1[i] < 0.8 else 1 - y[i]
    x1 += np.random.normal(loc=0, scale=0.1, size=(1000))  # Add noise

    # TODO: Optimize
    x2 = np.random.rand(1000)
    for i in range(len(x2)):
        x2[i] = y[i] if x2[i] < 0.5 else 1 - y[i]
    x2 += np.random.normal(loc=0, scale=0.1, size=(1000))  # Add noise

    dataset = np.zeros((1000, 3))
    dataset[:, 0] = x1
    dataset[:, 1] = x2
    dataset[:, 2] = y

    # print(len(np.where(dataset[:, 1] != dataset[:, 2])[0])) # Sanity check
    # print(len(np.where(dataset[:, 0] != dataset[:, 2])[0])) # Sanity check
    return dataset

def calculate_gini(histogram):
    pass

def create_histogram(X, feature_idx, bins = 10):
    feature = X[:, feature_idx]
    Y = X[:, -1]

    max = X.max()
    min = X.min()
    bin_edges = np.linspace(min, max, bins + 1)
    zeros = np.zeros(bins + 1)  # + 1 for anything > sample max
    ones = np.zeros(bins + 1)  # + 1 for anything > sample max

    for idx, f in enumerate(feature):
        # TODO(@motiwari): Change this to a binary search
        y = Y[idx]
        count_bucket = zeros if y == 0 else ones
        assigned = False
        for b_e_idx, b_e in enumerate(bin_edges):
            if f < b_e:  # Using < instead of <= prefers the right bucket
                count_bucket[b_e_idx] += 1
                assigned = True
                break
        if not assigned:
            count_bucket[bins] += 1  # Greater than sample max
    print(zeros)
    print(ones)
    

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
    X = create_data()
    ground_truth(X, show=False)
    create_histogram(X, 0)


if __name__ == "__main__":
    np.random.seed(1)
    np.set_printoptions(threshold=sys.maxsize)
    main()
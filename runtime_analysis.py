import numpy as np
import matplotlib.pyplot as plt

from utils.data_generator import create_data
from data_structures.tree import Tree
from utils.utils import class_to_idx


def analyze_runtime(verbose=True, take_log=True, show=True) -> None:
    """
    Sequentially creates toy binary trees that get larger and computes the
    normalized complexity (i.e. queries per node) of the trees as they get larger (by base^exponent)
    of the original dataset size "starting_size"

    :param verbose: if True, the function prints num_splits and num_queries
    :param take_log: whether to take the logarithm of the dataset sizes
    :param take_log: whether to plot the results (True) or save the figure (False)
    :return: None but plots the normalized complexity
    """
    # modify these params to test datasets scaled differently
    base = 10
    max_exponent = 5
    starting_size = 100
    num_trials = 10
    max_depth = 1
    np.random.seed(0)

    avg_num_queries = np.empty((max_exponent, num_trials))

    for i in range(max_exponent):
        for trial in range(num_trials):
            X = create_data(starting_size * pow(base, i))
            data = X[:, :-1]
            labels = X[:, -1]
            classes_arr = np.unique(labels)
            classes = class_to_idx(classes_arr)

            t = Tree(data=data, labels=labels, max_depth=max_depth, classes=classes, min_impurity_decrease=1e-3)
            t.fit()
            avg_num_queries[i, trial] = t.num_queries / t.num_splits

            if verbose:
                print(
                    "=> built trees with N:",
                    starting_size * pow(base, i),
                    "trial:",
                    trial,
                )
                print("\n")

    sizes = [starting_size * pow(base, i) for i in range(max_exponent)]
    if take_log:
        plt.plot(
            np.log10(sizes),
            np.mean(avg_num_queries, axis=1),
            color="r",
            label="normalized queries",
        )
        plt.xlabel("log$N$")
        plt.ylabel("Average number of datapoints used per split")
        plt.show() if show else plt.savefig("log_queries.png")

    else:
        plt.plot(
            sizes,
            np.mean(avg_num_queries, axis=1),
            color="r",
            label="normalized queries",
        )
        plt.xlabel("$N$")
        plt.ylabel("Average number of datapoints used per split")
        plt.show() if show else plt.savefig("queries.png")


def main():
    analyze_runtime()


if __name__ == "__main__":
    main()

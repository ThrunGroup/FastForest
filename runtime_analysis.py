import numpy as np
import matplotlib.pyplot as plt

from tests.data_generator import create_data
from data_structures.tree import Tree
from utils.utils import class_to_idx


def analyze_runtime(verbose=True, take_log=False) -> None:
    """
    Sequentially creates toy binary trees that get larger and computes the
    normalized complexity (i.e. queries per node) of the trees as they get larger (by base^exponent)
    of the original dataset size "starting_size"

    :param verbose: if True, the function prints num_splits and num_queries
    :return: None but plots the normalized complexity
    """
    # modify these params to test datasets scaled differently
    base = 10
    max_exponent = 5
    starting_size = 100
    num_trials = 50
    max_depth = 10
    np.random.seed(0)
    avg_norm_queries = []

    for i in range(max_exponent):
        norm_queries = 0
        total_queries = 0
        total_splits = 0
        for trial in range(num_trials):
            X = create_data(starting_size * pow(base, i))
            data = X[:, :-1]
            labels = X[:, -1]
            classes_arr = np.unique(labels)
            classes = class_to_idx(classes_arr)

            t = Tree(data=data, labels=labels, max_depth=max_depth, classes=classes)
            t.fit()

            total_queries += t.num_queries
            total_splits += t.num_splits
            norm_queries += t.num_queries / t.num_splits

        avg_norm_queries.append(norm_queries / num_trials)
        if verbose:
            print("=> built trees with datapoints", starting_size * pow(base, i))
            print("=> --total queries:", total_queries / num_trials)
            print("=> --total splits:", total_splits / num_trials)
            print("\n")

    sizes = [starting_size * pow(base, i) for i in range(len(avg_norm_queries))]
    if take_log:
        plt.plot(
            np.log10(sizes), avg_norm_queries, color="r", label="normalized queries"
        )
        plt.xlabel("log$N$")
        plt.ylabel("Average number of datapoints used per split")
        plt.savefig("log_queries.png")
    else:
        plt.plot(sizes, avg_norm_queries, color="r", label="normalized queries")
        plt.xlabel("$N$")
        plt.ylabel("Average number of datapoints used per split")
        plt.savefig("queries.png")


def main():
    analyze_runtime()


if __name__ == "__main__":
    main()

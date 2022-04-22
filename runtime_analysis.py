import numpy as np
import sys
import matplotlib.pyplot as plt
import math

from typing import Tuple
from data_generator import create_data
from mab_functions import solve_mab
from tree import Tree
from forest import Forest
from utils import class_to_idx


def test_binary_forest_time(verbose=True) -> None:
    """
    Sequentially creates toy binary trees that get larger and computes the
    normalized complexity (i.e. queries per node) of the trees as they get larger (by base^exponent)
    of the original dataset size "starting_size"
    :param verbose: if True, the function prints num_splits and num_queries
    :return: None but plots the normalized complexity
    """
    # modify these params to test datasets scaled differently
    sizes = 10
    max_exponent = 3
    starting_size = 100
    num_trials = 50
    max_depth = 10
    avg_norm_queries = []
    for i in range(max_exponent):
        norm_queries = 0
        total_queries = 0
        total_splits = 0

        for trial in range(num_trials):
            # create the dataset
            X = create_data(starting_size * pow(sizes, i))
            data = X[:, :-1]
            labels = X[:, -1]
            classes_arr = np.unique(labels)
            classes = class_to_idx(classes_arr)

            t = Tree(data=data, labels=labels, max_depth=max_depth, classes=classes)
            t.fit()

            total_queries += t.num_queries
            total_splits += t.num_splits
            norm_queries += (t.num_queries / t.num_splits)

        avg_norm_queries.append(norm_queries / num_trials)
        if verbose:
            print("=> built trees with datapoints", starting_size * pow(sizes, i))
            print("=> --total queries:", total_queries/num_trials)
            print("=> --total splits:", total_splits/num_trials)
            print("=> --max depth:", max_depth)
            print("\n")

    base = [starting_size * pow(sizes, i) for i in range(len(avg_norm_queries))]
    plt.plot(base, avg_norm_queries, color='r', label='normalized queries')
    plt.xlabel("Number of Datapoints")
    plt.ylabel("Number of Queries")
    
    print("saving figure")
    plt.savefig('norm_queries.png')


def main():
    print("Testing toy forest runtime")
    test_binary_forest_time()


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    np.random.seed(0)
    main()
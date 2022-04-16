import numpy as np
import sys
import matplotlib.pyplot as plt
import math

from typing import Tuple
from data_generator import create_data
from mab_functions import solve_mab
from tree import Tree
from forest import Forest


def test_binary_forest_time() -> None:
    num_trials = 50
    avg_query = []
    for i in range(5):
        queries = 0
        print("=> starting trees with datapoints", 10000 * pow(2, i))
        for trial in range(num_trials):
            # create the dataset
            X = create_data(10000 * pow(2, i))
            data = X[:, :-1]
            labels = X[:, -1]

            # create tree and fit
            t = Tree(data=data, labels=labels, max_depth=5)
            t.fit()
            queries += t.num_queries
        avg_query.append(queries/num_trials)

    plt.plot(avg_query)
    plt.savefig('runtime plot.png')


def main():
    print("Testing toy forest runtime")
    test_binary_forest_time()

    """
    print("\n" * 4)
    print("Testing forest digit dataset agreement:")
    test_forest_digits()
    """


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    np.random.seed(0)
    main()
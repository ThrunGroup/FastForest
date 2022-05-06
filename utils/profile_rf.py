import cProfile as profile
import pstats
import argparse
import os
import numpy as np
import sklearn.datasets
from sklearn.ensemble import RandomForestClassifier

from data_structures.forest_classifier import ForestClassifier
from utils.constants import IRIS, DIGITS, FASTFOREST, SKLEARN


def main() -> None:
    """
    main() provides profiling of random forest algorithms with different dataset.
    The statistics of the profile are stored in a file.
    """
    args = parse_args()
    np.random.seed(args.seed)
    prof = profile.Profile()
    prof.enable()
    fit_forest(args)
    prof.disable()

    stats = pstats.Stats(prof).strip_dirs().sort_stats("tottime")
    filename = os.path.join("profiles", get_filename(args))
    if not os.path.exists("profiles"):
        os.makedirs("profiles")
    stats.dump_stats(filename)
    if args.verbose:
        stats.print_stats(20)


def fit_forest(args: argparse.Namespace) -> None:
    """
    For specific RF algorithm and dataset given in "args", run the alogorithm to fit to the dataset.
    :param args: args is an object of argparse.Namespace that contains variables needed for an experiment
    """
    if args.dataset == IRIS:
        iris = sklearn.datasets.load_iris()
        data, labels = iris.data, iris.target
    elif args.dataset == DIGITS:
        digits = sklearn.datasets.load_digits()
        data, labels = digits.data, digits.target
    else:
        raise Exception("Invalid choice of dataset")

    if args.algorithm == FASTFOREST:
        forest = Forest(
            data=data, labels=labels, n_estimators=args.n_estimators, max_depth=5
        )
        forest.fit()
    elif args.algorithm == SKLEARN:
        forest = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=5)
        forest.fit(data, labels)
    else:
        raise Exception("Invalid random foreset algorithm")


def get_filename(args):
    """
    Create the filename suffix for an experiment, given its configuration.
    :param args: args is an object of argparse.Namespace that contains variables needed for an experiment
    """
    return (
        "-a-"
        + args.algorithm
        + "-n-"
        + str(args.n_estimators)
        + "-d-"
        + str(args.dataset)
        + "-s-"
        + str(args.seed)
    )


def parse_args() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        type=str,
        default=FASTFOREST,
        choices=[FASTFOREST, SKLEARN],
        help="Random forest algorithm name (default: FASTFOREST)",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=1,
        help="Number of tree estimators in a random forest (default: 1)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=IRIS,
        choices=[IRIS, DIGITS],
        help="data name (default: IRIS)",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="Whether to print profile results (default: True)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

import cProfile as profile
import pstats
import argparse
import os
import numpy as np
import sklearn.datasets

from forest import Forest
from sklearn.ensemble import RandomForestClassifier


def main():
    args = parse_args()
    np.random.seed(args.seed)
    prof = profile.Profile()

    # Profile starts
    prof.enable()
    fit_forest(args)
    prof.disable()
    # Profile ends

    # Store profile in a file
    stats = pstats.Stats(prof).strip_dirs().sort_stats("tottime")
    filename = os.path.join("profile", get_file_name(args))
    if not os.path.exists("profile"):
        os.makedirs("profile")
    stats.dump_stats(filename)
    if args.verbose:
        stats.print_stats(20)


def fit_forest(args):
    if args.data == "IRIS":
        iris = sklearn.datasets.load_iris()
        data, labels = iris.data, iris.target
    elif args.data == "DIGITS":
        digits = sklearn.datasets.load_digits()
        data, labels = digits.data, digits.target
    else:
        raise Exception("Wrong dataset")

    if args.algorithm == "FASTFOREST":
        forest = Forest(
            data=data, labels=labels, n_estimators=args.n_estimators, max_depth=5
        )
        forest.fit()  # Jay: Useful to make the "fit" method of "Forest" class takes same inputs as one of SKLEARN?
    elif args.algorithm == "SKLEARN":
        forest = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=5)
        forest.fit(data, labels)
    else:
        raise Exception("Wrong random foreset algorithm")


def get_file_name(args):
    """
    Create the filename suffix for an experiment, given its configuration.
    """
    return (
        "-a-"
        + args.algorithm
        + "-n-"
        + str(args.n_estimators)
        + "-d-"
        + str(args.data)
        + "-s-"
        + str(args.seed)
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        type=str,
        default="FASTFOREST",
        choices=["FASTFOREST", "SKLEARN"],
        help="Augmentation optimizer name (default: FASTFOREST)",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=1,
        help="Number of tree estimators in a random forest (default: 1)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="IRIS",
        choices=["IRIS", "DIGIT"],
        help="data name (default: IRIS)",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="Whether to print profile results (default: True)",
    )
    print(parser.parse_args())
    return parser.parse_args()


if __name__ == "__main__":
    main()

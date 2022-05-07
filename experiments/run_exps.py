"""
This script contains scaffolding to run many experiments, listed in a config
file such as auto_exp_config.py.

This script will parse each line (= exp configuration) from the config file and
run the corresponding experiment. It can also run many experiments in parallel
by using the pool.apply_async calls instead of the explicit run_exp calls.
"""

import importlib
import multiprocessing as mp
import numpy as np
import traceback
import copy
import argparse
import sklearn.datasets
import os
from typing import List

from data_structures.forest_classifier import ForestClassifier
from utils.constants import MAB, EXACT
from create_config import dict_to_str


def get_args() -> argparse.Namespace:
    """
    Gather the experimental arguments from the command line.

    :return: Namespace object from argparse containing each argument
    """
    parser = argparse.ArgumentParser(description="Solve Forest Problems")
    parser.add_argument(
        "-t",
        "--task",
        help="Classification (C) or Regression (R)",
        default="C",
        type=str,
    )
    parser.add_argument(
        "-x",
        "--fixed",
        help="Variable to keep fixed: budget (B) or performance (P)",
        default="B",
        type=str,
    )
    parser.add_argument(
        "-v",
        "--value",
        help="Metric value. For performance metric, the metric we should reach. For budget, the budget value",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "-s",
        "--solver",
        help="Solver (algorithm) to use",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--parameters",
        help="Algorithm parameters. Meant to be a dict object",
        default={},
        type=dict,
    )

    # Batch experiment configuration
    parser.add_argument(
        "-e",
        "--exp_config",
        help="What experiment configuration to run",
        default="auto_exp_config.py",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--force",
        help="Whether to force rerun experiments",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-u",
        "--use_mp",
        help="Whether to use multiprocessing to run multiple experiments in parallel",
        default=False,
        type=bool,
    )
    return parser.parse_args()


def remap_args(args: argparse.Namespace, exp: List):
    """
    Parses an experiment config line (a list) into an args variable (a Namespace).

    :param args: Namespace object whose args are to be modified
    :param exp: Experiment configuration (list of arguments)
    """
    args.task = exp[0]
    args.force = exp[1]
    args.value = exp[2]
    args.solver = exp[3]
    args.parameters = exp[4]
    # Do not re-assign args.force or args.use_mp as that's passed at the batch level
    return args


def get_logfile_name(args: argparse.Namespace) -> str:
    """
    Returns the name of the logfile with the given arguments.

    :param args: arguments of the experiment
    :return: string representing the logfile's name
    """
    return os.path.join(
        "logs",
        "t-"
        + str(args.task)
        + "-v-"
        + str(args.value)
        + "-a-"
        + str(args.solver)
        + "-p-"
        + dict_to_str(args.parameters)
        + ".txt",
    )


def run_exp(args: argparse.Namespace, logfile: str) -> None:
    iris = sklearn.datasets.load_iris()
    data, labels = iris.data, iris.target

    if args.solver == MAB:
        FC = ForestClassifier(
            data=data, labels=labels, max_depth=5, budget=50, solver=MAB
        )
    elif args.solver == EXACT:
        FC = ForestClassifier(
            data=data, labels=labels, max_depth=5, budget=50, solver=EXACT
        )
    else:
        print(args.solver)
        raise Exception("Invalid algorithm specified")

    FC.fit()
    print("Forest number of queries:", FC.num_queries)
    acc = np.sum(FC.predict_batch(data)[0] == labels)

    with open(logfile, "w+") as fout:
        fout.write(
            "Fixed: "
            + ("Budget " if args.fixed == "B" else "Metric ")
            + "\n"
            + "Fixed Value:"
            + str(acc)
            + "\n"
        )
        fout.write(
            "Measured: "
            + ("Metric " if args.fixed == "B" else "Budget ")
            + "\n"
            + "Measured value: "
            + str(FC.num_queries)
        )


def main() -> None:
    """
    Run all the experiments in the experiments lists specified by the -e
    argument, and write the final results (including logstrings) to files. Can
    run multiple experiments in parallel by using the pool.apply_async calls
    below instead of the explicit run_exp calls.
    """
    args = get_args()  # Uses default values for now as placeholder to instantiate args
    imported_config = importlib.import_module(args.exp_config.strip(".py"))

    if args.use_mp:
        pool = mp.Pool()

    for exp in imported_config.experiments:
        args = remap_args(args, exp)
        logfile = os.path.join(get_logfile_name(args))
        if os.path.exists(logfile) and not args.force:
            print("Warning: already have data for experiment", logfile)
            continue
        else:
            print("Running exp:", logfile)

        """
        WARNING: The apply_async calls below are NOT threadsafe. In particular,
        strings in python are lists, which means they are passed by reference.
        This means that if a NEW thread gets the SAME reference as the other
        threads, and updates the object, the OLD thread will write to the wrong
        file. Therefore, whenever using multiprocessing, need to copy.deepcopy()
        all the arguments. Don't need to do this for the explicit run_exp calls
        though since those references are used appropriately (executed
        sequentially)
        """
        try:
            if args.use_mp:
                pool.apply_async(
                    run_exp,
                    args=(copy.deepcopy(args), copy.deepcopy(logfile)),
                )  # Copy inline to copy OTF
            else:
                run_exp(args, logfile)
        except Exception as _e:
            print(traceback.format_exc())

    if args.use_mp:
        pool.close()
        pool.join()
    print("Finished")


if __name__ == "__main__":
    main()

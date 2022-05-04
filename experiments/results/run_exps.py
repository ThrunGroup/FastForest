"""
This script contains scaffolding to run many experiments, listed in a config
file such as auto_exp_config.py.

This script will parse each line (= exp configuration) from the config file and
run the corresponding experiment. It can also run many experiments in parallel
by using the pool.apply_async calls instead of the explicit run_exp calls.
"""

import importlib
import multiprocessing as mp
import traceback
import os
import argparse
import numpy as np


def run_exp(args: argparse.Namespace, method_name: str, mnist_imgs: np.ndarray) -> None:
    """
    Runs an experiment with the given parameters, and writes the results to the
    files. I seem to have modified this to play around with MNIST images?
    It looks like it can be used to generate random data by uncommenting/commenting some of the lines

    :param args: arguments of experiments to run
    :param method_name: which method to call, bandit_matching_pursuit or matching_pursuit
    :param mnist_imgs: numpy array of MNIST images
    """
    np.random.seed(args.seed)
    # data = 2 * np.random.rand(args.N + 1, args.d) - 1  # + 1 to generate the signal

    data = np.transpose(mnist_imgs) / 255
    data = data[
        np.random.choice(range(784), args.N + 1, replace=False)
    ]  # Select random dimensions
    data = data[
        :, np.random.choice(range(60000), args.d, replace=False)
    ]  # + 1 to generate the signal

    data /= np.linalg.norm(data, axis=1).reshape(-1, 1)  # Normalize signal too?
    signal = data[0]
    atoms = data[1:]
    method_name(input_signal=signal, atoms=atoms, args=args)


def main() -> None:
    """
    Run all the experiments in the experiments lists specified by the -e
    argument, and write the final results (including logstrings) to files. Can
    run multiple experiments in parallel by using the pool.apply_async calls
    below instead of the explicit run_exp calls.
    """
    args = get_args()  # Uses default values for now as placeholder to instantiate args
    imported_config = importlib.import_module(args.exp_config.strip(".py"))
    mnist_imgs = get_minst_data()

    pool = mp.Pool()
    for exp in imported_config.experiments:
        args = remap_args(args, exp)
        bandit = exp[0] == "B_MP"
        logfile = os.path.join(get_logfile_name(args, bandit=bandit))
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
            if exp[0] == "MP":
                # pool.apply_async(run_exp, args=(copy.deepcopy(args), matching_pursuit, copy.deepcopy(logfile)+'-profile'))  # Copy inline to copy OTF
                run_exp(args, matching_pursuit, mnist_imgs)
            elif exp[0] == "B_MP":
                # pool.apply_async(run_exp, args=(copy.deepcopy(args), bandit_matching_pursuit, copy.deepcopy(logfile)+'-profile-b'))  # Copy inline to copy OTF
                run_exp(args, bandit_matching_pursuit, mnist_imgs)
            else:
                raise Exception("Invalid algorithm specified")
        except Exception as _e:
            print(traceback.format_exc())

    pool.close()
    pool.join()
    print("Finished")


if __name__ == "__main__":
    main()

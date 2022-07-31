import os
import numpy as np

from experiments.runtime_exps.compare_runtimes import (
    compare_runtimes,
    make_regression,
)

from utils.constants import MNIST_STR
from experiments.exp_constants import SCALING_NUM_SEEDS
from experiments.datasets import data_loader


def main(is_classification=True):
    size_to_insertions_dict = {}
    size_to_time_dict = {}
    if is_classification:
        train_data, train_labels, _, _ = data_loader.fetch_data(MNIST_STR)
        filename_insertion = "size_to_insertions_dict"
        filename_time = "size_to_time_dict"
        models = ["HRFC"]  # , "ERFC", "HRPC"]
        subsample_size_list = [
            1000,
            3000,
            5000,
            10000,
            15000,
            20000,
            30000,
            35000,
            40000,
            50000,
            60000,
        ]
    else:
        train_data, train_labels = make_regression(
            200000, n_features=50, n_informative=5, random_state=0
        )
        filename_insertion = "size_to_insertions_dict_regression"
        filename_time = "size_to_time_dict_regression"
        models = ["HRFR"]  # , "ERFR", "HRPR"]
        subsample_size_list = [
            5000,
            10000,
            20000,
            40000,
            60000,
            80000,
            100000,
            120000,
            160000,
            200000,
        ]
    for model in models:
        for C_SUBSAMPLE_SIZE in subsample_size_list:
            print("\n\n")
            num_queries = 0.0
            run_time = 0.0
            num_trials = 0
            for fitting_seed in range(SCALING_NUM_SEEDS):
                np.random.seed(fitting_seed)
                rng = np.random.default_rng(fitting_seed)
                idcs = rng.choice(len(train_data), size=C_SUBSAMPLE_SIZE, replace=True)
                train_data_subsampled = train_data[idcs]
                train_labels_subsampled = train_labels[idcs]
                results = compare_runtimes(
                    dataset_name=MNIST_STR,
                    compare=model,
                    train_data=train_data_subsampled,
                    train_targets=train_labels_subsampled,
                    predict=False,
                    run_theirs=False,
                    filename=model
                    + "_"
                    + str(C_SUBSAMPLE_SIZE)
                    + "_profile_"
                    + str(fitting_seed),
                )
                num_queries += np.mean(np.array(results["our_num_queries"]))
                run_time += np.mean(np.array(results["our_train_times"]))
                num_trials += 1
            num_queries /= num_trials
            run_time /= num_trials
            size_to_insertions_dict[C_SUBSAMPLE_SIZE] = num_queries
            size_to_time_dict[C_SUBSAMPLE_SIZE] = run_time
        with open(model + "_" + filename_insertion, "w+") as fout:
            fout.write(str(size_to_insertions_dict))
        with open(model + "_" + filename_time, "w+") as fout:
            fout.write(str(size_to_time_dict))


if __name__ == "__main__":
    main(is_classification=True)
    main(is_classification=False)

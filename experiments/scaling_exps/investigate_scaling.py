import os
import numpy as np
import pandas as pd
from collections import defaultdict

from experiments.runtime_exps.compare_runtimes import (
    compare_runtimes,
    make_regression,
)

from utils.constants import MNIST_STR
from experiments.exp_constants import SCALING_NUM_SEEDS
from experiments.datasets import data_loader


def main(is_classification=True):
    log_data = defaultdict(list)
    if is_classification:
        train_data, train_labels, _, _ = data_loader.fetch_data(MNIST_STR)
        filename = "scaling_classification"
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
        filename = "scaling_regression"
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
            num_queries = []
            run_time = []
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
                    max_depth=5,
                )
                num_queries.append(np.mean(np.array(results["our_num_queries"])))
                run_time.append(np.array(results["our_train_times"]))
                num_trials += 1
            avg_num_queries = np.mean(num_queries)
            avg_run_time = np.mean(run_time)
            std_num_queries = np.std(num_queries) / np.sqrt(num_trials)
            std_run_time = np.std(run_time) / np.sqrt(num_trials)
            log_data["size"].append(C_SUBSAMPLE_SIZE)
            log_data["avg_num_queries"].append(avg_num_queries)
            log_data["avg_run_time"].append(avg_run_time)
            log_data["std_num_queries"].append(std_num_queries)
            log_data["std_run_time"].append(std_run_time)
        log_data_df = pd.DataFrame(log_data)
        log_data_df.to_csv(model + "_" + filename, index=False)


if __name__ == "__main__":
    main(is_classification=True)
    main(is_classification=False)

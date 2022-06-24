from mnist import MNIST
from sklearn.datasets import make_regression
import numpy as np
import sys

print(sys.path)
sys.path.append("C:\\Users\\MSI\\Desktop\\FastForest\\FastForest")

from experiments.runtime_exps.compare_runtimes import *


def main(is_classification=True):
    if is_classification:
        mndata = MNIST(os.path.join("..", "mnist"))

        train_data, train_labels = mndata.load_training()
        size_to_time_dict = {}
        filename = "size_to_time_dict"
        model = "HRFC"
        subsample_size_list = [
            1,
            40000,
            80000,
            160000,
            320000,
            640000,
        ]
    else:
        train_data, train_labels = make_regression(
            200000, n_features=50, n_informative=5, random_state=0
        )
        size_to_time_dict = {}
        filename = "size_to_time_dict_regression"
        model = "HRFR"
        subsample_size_list = [
            1,
            20000,
            40000,
            80000,
            160000,
            320000,
        ]
    for C_SUBSAMPLE_SIZE in subsample_size_list:
        print("\n\n")
        run_time = 0.0
        num_trials = 0
        for fitting_seed in range(0, 5):
            np.random.seed(fitting_seed)
            rng = np.random.default_rng(fitting_seed)
            idcs = rng.choice(len(train_data), size=C_SUBSAMPLE_SIZE, replace=True)
            train_data_subsampled = np.array(train_data)[idcs]
            train_labels_subsampled = np.array(train_labels)[idcs]
            results = compare_runtimes(
                model,
                train_data_subsampled,
                train_labels_subsampled,
                predict=False,
                run_theirs=False,
                filename=model
                + "_"
                + str(C_SUBSAMPLE_SIZE)
                + "_profile_"
                + str(fitting_seed),
            )
            run_time += np.mean(np.array(results["our_train_times"]))
            num_trials += 1
        run_time /= num_trials
        size_to_time_dict[C_SUBSAMPLE_SIZE] = run_time
    with open(filename, "w+") as fout:
        fout.write(str(size_to_time_dict))

        # compare_runtimes(
        #     "ERFC",
        #     train_images_subsampled,
        #     train_labels_subsampled,
        #     test_images,
        #     test_labels,
        #     run_theirs=False,
        #     profile_name="ERFC_" + str(C_SUBSAMPLE_SIZE) + "_profile",
        # )
        #
        # compare_runtimes(
        #     "HRPC",
        #     train_images_subsampled,
        #     train_labels_subsampled,
        #     test_images,
        #     test_labels,
        #     run_theirs=False,
        #     profile_name="HRPC_" + str(C_SUBSAMPLE_SIZE) + "_profile",
        # )


if __name__ == "__main__":
    main(is_classification=True)
    main(is_classification=False)

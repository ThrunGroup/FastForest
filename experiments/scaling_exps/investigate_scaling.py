from mnist import MNIST
import numpy as np
import sys

print(sys.path)
sys.path.append('C:\\Users\\MSI\\Desktop\\FastForest\\FastForest')

from experiments.runtime_exps.compare_runtimes import *


def main():
    mndata = MNIST(os.path.join("..", "mnist"))

    train_images, train_labels = mndata.load_training()
    size_to_time_dict = {}
    for C_SUBSAMPLE_SIZE in [
        10000,
        80000,
        160000,
        320000,
        640000,
        1280000,
    ]:
        print("\n\n")
        run_time = .0
        num_trials = 0
        for fitting_seed in range(0, 5):
            np.random.seed(fitting_seed)
            rng = np.random.default_rng(fitting_seed)
            idcs = rng.choice(60000, size=C_SUBSAMPLE_SIZE, replace=True)
            train_images_subsampled = np.array(train_images)[idcs]
            train_labels_subsampled = np.array(train_labels)[idcs]
            results = compare_runtimes(
                "HRFC",
                train_images_subsampled,
                train_labels_subsampled,
                predict=False,
                run_theirs=False,
                filename="HRFC_"
                + str(C_SUBSAMPLE_SIZE)
                + "_profile_"
                + str(fitting_seed),
            )
            run_time += np.mean(np.array(results["our_train_times"]))
            num_trials += 1
        run_time /= num_trials
        size_to_time_dict[C_SUBSAMPLE_SIZE] = run_time
    with open("size_to_time_dict", "w+") as fout:
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
    main()

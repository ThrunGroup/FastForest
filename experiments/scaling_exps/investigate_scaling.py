from mnist import MNIST

from experiments.runtime_exps.compare_runtimes import *

import pprint


def main():
    mndata = MNIST("mnist/")

    train_images, train_labels = mndata.load_training()

    # For accuracy comparison. Looks ok
    # test_images, test_labels = mndata.load_testing()
    # test_images = np.array(test_images)
    # test_labels = np.array(test_labels)

    for C_SUBSAMPLE_SIZE in [
        10000,
        20000,
        30000,
        40000,
        50000,
        60000,
        80000,
        100000,
        120000,
        140000,
        160000,
        180000,
        200000,
        220000,
        240000,
        260000,
        280000,
        300000,
        320000,
    ]:
        print("\n\n")
        for fitting_seed in range(100, 105):
            np.random.seed(fitting_seed)
            rng = np.random.default_rng(fitting_seed)
            idcs = rng.choice(60000, size=C_SUBSAMPLE_SIZE, replace=True)
            train_images_subsampled = np.array(train_images)[idcs]
            train_labels_subsampled = np.array(train_labels)[idcs]
            compare_runtimes(
                "HRFC",
                full_train_data=train_images,
                full_train_targets=train_labels,
                test_data=None,
                test_targets=None,
                starting_seed=100,
                num_seeds=20,
                predict=False,
                run_theirs=False,
                profile_name="HRFC_" + str(C_SUBSAMPLE_SIZE) + "_profile_",
                C_SUBSAMPLE_SIZE=C_SUBSAMPLE_SIZE,
            )
        )
        print("Results for ", C_SUBSAMPLE_SIZE, " above")

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

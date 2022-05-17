from mnist import MNIST

from experiments.compare_runtimes import *


def main():
    mndata = MNIST("mnist/")

    train_images, train_labels = mndata.load_training()

    for C_SUBSAMPLE_SIZE in [
        5000,
        10000,
        20000,
        40000,
        60000,
        # 80000,
        # 160000,
        # 320000,
        # 640000,
        # 1280000,
    ]:
        print("\n\n")

        print(
            compare_runtimes(
                "HRFC",
                full_train_data=train_images,
                full_train_targets=train_labels,
                full_test_data=None,
                full_test_targets=None,
                starting_seed=100,
                num_seeds=5,
                predict=False,
                run_theirs=False,
                profile_name="HRFC_" + str(C_SUBSAMPLE_SIZE) + "_profile_",
                C_SUBSAMPLE_SIZE=C_SUBSAMPLE_SIZE,
            )
        )

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

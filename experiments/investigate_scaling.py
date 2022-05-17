from mnist import MNIST

from experiments.compare_runtimes import *


def main():
    mndata = MNIST("mnist/")

    train_images, train_labels = mndata.load_training()

    for C_SUBSAMPLE_SIZE in [
        80000,
        # 160000,
        320000,
        # 640000,
        # 1280000,
    ]:
        for fitting_seed in range(100, 101):
            np.random.seed(fitting_seed)
            idcs = np.random.choice(60000, size=C_SUBSAMPLE_SIZE, replace=True)
            train_images_subsampled = np.array(train_images)[idcs]
            train_labels_subsampled = np.array(train_labels)[idcs]
            compare_runtimes(
                "HRFC",
                train_images_subsampled,
                train_labels_subsampled,
                predict=False,
                run_theirs=False,
                profile_name="HRFC_"
                + str(C_SUBSAMPLE_SIZE)
                + "_profile_"
                + str(fitting_seed),
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

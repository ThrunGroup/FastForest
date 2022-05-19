from mnist import MNIST

from experiments.runtime_exps.compare_runtimes import *


def main():
    mndata = MNIST(os.path.join("..", "mnist"))

    train_images, train_labels = mndata.load_training()

    for C_SUBSAMPLE_SIZE in [
        5000,
        10000,
        15000,
        20000,
        25000,
        30000,
        35000,
        40000,
        45000,
        50000,
        55000,
        60000,
        80000,
        160000,
        320000,
    ]:
        print("\n\n")
        for fitting_seed in range(100, 120):
            np.random.seed(fitting_seed)
            rng = np.random.default_rng(fitting_seed)
            idcs = rng.choice(60000, size=C_SUBSAMPLE_SIZE, replace=True)
            train_images_subsampled = np.array(train_images)[idcs]
            train_labels_subsampled = np.array(train_labels)[idcs]
            compare_runtimes(
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
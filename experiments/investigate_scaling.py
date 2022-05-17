from mnist import MNIST

from experiments.compare_runtimes import *

import pprint


def main():
    mndata = MNIST("mnist/")

    train_images, train_labels = mndata.load_training()

    # For accuracy comparison. Looks ok
    # test_images, test_labels = mndata.load_testing()
    # test_images = np.array(test_images)
    # test_labels = np.array(test_labels)

    for C_SUBSAMPLE_SIZE in [
        # 100,
        # 200,
        # 1000,
        # 5000,
        # 10000,
        # 15000,
        # 20000,
        # 25000,
        # 30000,
        # 35000,
        # 40000,
        # 45000,
        # 50000,
        # 55000,
        # 60000,
        # 80000,
        # 160000,
        # 320000,
        # 640000,
        # 1280000,
    ]:
        print("\n\n")
        pp = pprint.PrettyPrinter(indent=2)

        pp.pprint(
            compare_runtimes(
                "HRFC",
                full_train_data=train_images,
                full_train_targets=train_labels,
                test_data=None,
                test_targets=None,
                starting_seed=100,
                num_seeds=10,
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

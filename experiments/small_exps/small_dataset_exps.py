"""
Contains a quick and dirty script to compute when MABSplit outperforms RF in terms of sample complexity and wall clock time
to respond to R3
"""

from experiments.runtime_exps.compare_runtimes import *

import matplotlib.pyplot as plt
import ast

MIN_DATA_SIZE = 1000
MAX_DATA_SIZE = 1801
INTERVAL = 100
NUM_BINS = int((MAX_DATA_SIZE - MIN_DATA_SIZE)/INTERVAL) + 1  # Rounds down, NOT banker's rounding
EXP_SIZES = range(MIN_DATA_SIZE, MAX_DATA_SIZE, INTERVAL)
FIGSIZE = (5, 3)

def make_plots():
    queries = np.zeros((2, NUM_BINS))
    queries_std = np.zeros((2, NUM_BINS))

    runtimes = np.zeros((2, NUM_BINS))
    runtimes_std = np.zeros((2, NUM_BINS))

    for dataset_size_idx, dataset_size in enumerate(EXP_SIZES):
        filename = MNIST_STR + "_HRFC_dict_intercept_" + str(dataset_size)
        with open(os.path.join("logs", filename)) as fin:
            log_dict = ast.literal_eval(fin.read())
            queries[0, dataset_size_idx] = log_dict['our_avg_num_queries']
            queries_std[0, dataset_size_idx] = log_dict['our_std_num_queries']
            queries[1, dataset_size_idx] = log_dict['their_avg_num_queries']
            queries_std[1, dataset_size_idx] = log_dict['their_std_num_queries']

            runtimes[0, dataset_size_idx] = log_dict['our_avg_train_time']
            runtimes_std[0, dataset_size_idx] = log_dict['our_std_train_time']
            runtimes[1, dataset_size_idx] = log_dict['their_avg_train_time']
            runtimes_std[1, dataset_size_idx] = log_dict['their_std_train_time']

    print(queries)
    print(runtimes)

    plt.figure(figsize=FIGSIZE)
    plt.title("RF vs. RF + MABSplit Sample Complexity")
    plt.ylabel("Number of samples drawn (sample complexity)")
    plt.xlabel("Subset size ($N$)")
    plt.plot(
        EXP_SIZES,
        queries[1],
        "rx",
        label="RF",
    )
    plt.plot(
        EXP_SIZES,
        queries[0],
        "bo",
        label="RF + MABSplit",
    )
    plt.errorbar(EXP_SIZES, queries[0], yerr=queries_std[0], barsabove=True, capsize=5)
    plt.errorbar(EXP_SIZES, queries[1], yerr=queries_std[1], barsabove=True, capsize=5)
    plt.legend()
    plt.show()

    plt.figure(figsize=FIGSIZE)
    plt.title("RF vs. RF + MABSplit Runtime")
    plt.ylabel("Wall clock time (s)")
    plt.xlabel("Subset size ($N$)")
    plt.plot(
        EXP_SIZES,
        runtimes[1],
        "rx",
        label="RF",
    )
    plt.plot(
        EXP_SIZES,
        runtimes[0],
        "bo",
        label="RF + MABSplit",
    )
    plt.errorbar(EXP_SIZES, runtimes[0], yerr=runtimes_std[0], barsabove=True, capsize=5)
    plt.errorbar(EXP_SIZES, runtimes[1], yerr=runtimes_std[1], barsabove=True, capsize=5)
    plt.legend()
    plt.show()



def run_exps():
    pp = pprint.PrettyPrinter(indent=2)
    train_images, train_labels, test_images, test_labels = data_loader.fetch_data(MNIST_STR)
    c_m = "HRFC"
    for dataset_size in range(1000, 3001, 100):
        print("Datasize size:", dataset_size)
        train_idcs = np.random.choice(train_images.shape[0], dataset_size, replace=False)
        train_images_subsampled = train_images[train_idcs]
        train_labels_subsampled = train_labels[train_idcs]
        pp.pprint(
            compare_runtimes(
                dataset_name=MNIST_STR,
                compare=c_m,
                train_data=train_images_subsampled,
                train_targets=train_labels_subsampled,
                original_test_data=None,
                test_targets=None,
                num_seeds=3,
                predict=False,
                run_theirs=True,
                filename=MNIST_STR + "_HRFC_dict_intercept_" + str(dataset_size),
                verbose=False,
                max_depth=None,
                max_leaf_nodes=None,
            )
        )


def main():
    # run_exps()
    make_plots()


if __name__ == '__main__':
    main()

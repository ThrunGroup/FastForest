import matplotlib.pyplot as plt
import numpy as np
import ast
import sys

from sklearn.linear_model import LinearRegression


def main(filename: str = "HRFC_size_to_time_dict"):
    with open(filename, "r+") as fin:
        size_to_time_dict = ast.literal_eval(fin.read())
        sizes = np.array(sorted(size_to_time_dict.keys()))
        counts = np.array([size_to_time_dict[k] for k in sizes])
    mean_counts = np.mean(counts)

    lr = LinearRegression()
    lr.fit(sizes.reshape(-1, 1), counts)

    log_lr = LinearRegression()
    log_lr.fit(np.log10(sizes).reshape(-1, 1), counts)

    hashes = np.linspace(1, np.max(sizes), 5000)
    hashes_predict = lr.predict(hashes.reshape(-1, 1))
    hashes_predict_log = log_lr.predict(np.log10(hashes).reshape(-1, 1))

    counts_predict = lr.predict(sizes.reshape(-1, 1))
    counts_predict_log = log_lr.predict(np.log10(sizes).reshape(-1, 1))
    hashes_R2 = 1 - (
        ((counts_predict.reshape(-1) - counts) ** 2).sum()
        / ((mean_counts - counts) ** 2).sum()
    )
    hashes_log_R2 = 1 - (
        ((counts_predict_log.reshape(-1) - counts) ** 2).sum()
        / ((mean_counts - counts) ** 2).sum()
    )

    print(hashes_R2, hashes_log_R2)

    plt.scatter(sizes, counts)
    plt.plot(
        hashes, hashes_predict, color="green", label=f"Linear fit, $R^2 = {hashes_R2}$"
    )
    plt.plot(
        hashes,
        hashes_predict_log,
        color="orange",
        label=f"Logarithmic fit fit, $R^2 = {hashes_log_R2}$",
    )
    dataset = "Regression" if "regression" in filename else "Classification"
    plt.ylabel("Number of minibatches queried by MABSplit")
    plt.xlabel(f"{dataset}: Dataset size")
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    main(filename="HRFC_size_to_time_dict")
    main(filename="HRFR_size_to_time_dict_regression")


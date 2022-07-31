import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
import sys

from sklearn.linear_model import LinearRegression


def main(filename: str = "HRFC_scaling_classification.csv"):
    df = pd.read_csv(filename)
    columns = ["num_queries", "run_time"]
    sizes = np.array(df["size"])
    for idx in range(2):
        counts = np.array(df["avg_" + columns[idx]])
        counts_err = np.array(df["std_" + columns[idx]])
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

        plt.errorbar(sizes, counts, yerr=counts_err, barsabove=True, capsize=5)
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
        if idx == 0:  # Number of queries comparison
            plt.ylabel("Number of histogram insertions queried by MABSplit")
        else:   # Runtime comparison
            plt.ylabel("Runtime(seconds) by MABSplit")
        plt.xlabel(f"{dataset}: Dataset size")
        plt.legend(loc="lower right")
        plt.show()


if __name__ == "__main__":
    main(filename="HRFC_scaling_classification")
    main(filename="HRFR_scaling_regression")


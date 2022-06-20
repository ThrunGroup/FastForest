import matplotlib.pyplot as plt
import numpy as np
import ast
from sklearn.linear_model import LinearRegression

# sizes = np.array(
#     [
#         5000,
#         10000,
#         15000,
#         20000,
#         25000,
#         30000,
#         35000,
#         40000,
#         45000,
#         50000,
#         55000,
#         60000,
#         80000,
#         160000,
#         320000,
#     ]
# )
#
# counts = np.array(
#     [
#         43.4,
#         80.7,
#         107.8,
#         146.7,
#         154,
#         207.6,
#         218.2,
#         229.9,
#         278.6,
#         275.5,
#         338.1,
#         379.1,
#         347.9,
#         413.2,
#         457.4,
#     ]
# )
with open("size_to_time_dict", "r+") as fin:
    size_to_time_dict = ast.literal_eval(fin.read())
    sizes = sorted(size_to_time_dict.keys())
    counts = [v for k, v in sizes]
mean_counts = np.mean(counts)

lr = LinearRegression()
lr.fit(sizes.reshape(-1, 1), counts)

log_lr = LinearRegression()
log_lr.fit(np.log10(sizes).reshape(-1, 1), counts)

hashes = np.linspace(5000, 325000, 5000)
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
plt.plot(hashes, hashes_predict, color="green", label="Linear fit, $R^2 = 0.58$")
plt.plot(
    hashes,
    hashes_predict_log,
    color="orange",
    label="Logarithmic fit fit, $R^2 = 0.93$",
)
plt.ylabel("Number of minibatches queried by MABSplit")
plt.xlabel("Dataset size")
plt.legend(loc="upper left")
plt.show()

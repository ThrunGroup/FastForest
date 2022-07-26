import numpy as np
import time

from data_structures.tree_classifier import TreeClassifier
from data_structures.forest_classifier import ForestClassifier


if __name__ == "__main__":
    data = np.zeros((10000, 2000))
    labels = np.zeros(10000)
    t = TreeClassifier(
        data=data,
        labels=labels,
        classes={0: 0},
        max_depth=1,
        make_discrete=True,
        bin_type="",
    )
    start_time = time.time()
    t.fit(data, labels)
    print(
        f"Time taken for fitting and creating discrete features: {time.time() - start_time}"
    )

    t.make_discrete = False
    start_time = time.time()
    t.fit(data, labels)
    print(f"Time taken only for fitting: {time.time() - start_time}")

    f = ForestClassifier(
        data=data,
        labels=labels,
        max_depth=1,
        classes={0: 0},
        make_discrete=True,
        bin_type="",
        n_estimators=10,
    )
    start_time = time.time()
    f.fit(data, labels)
    print(
        f"Time taken for fitting and creating discrete features: {time.time() - start_time}"
    )

    f.make_discrete = False
    start_time = time.time()
    f.fit(data, labels)
    print(f"Time taken only for fitting: {time.time() - start_time}")

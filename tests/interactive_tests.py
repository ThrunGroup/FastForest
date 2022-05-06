import sklearn.datasets
import numpy as np

from data_structures.tree_classifier import TreeClassifier
import utils.utils


def test_increasing_budget_tree_iris() -> None:
    iris = sklearn.datasets.load_iris()
    data, labels = iris.data, iris.target
    classes_arr = np.unique(labels)
    classes = utils.utils.class_to_idx(classes_arr)

    t1 = TreeClassifier(
        data=data, labels=labels, max_depth=5, classes=classes, budget=50
    )
    t1.fit()
    t1.tree_print()
    print("T1 Number of queries:", t1.num_queries)
    acc1 = np.sum(t1.predict_batch(data)[0] == labels)
    print()
    print()
    t2 = TreeClassifier(
        data=data, labels=labels, max_depth=5, classes=classes, budget=1000
    )
    t2.fit()
    t2.tree_print()

    print("T2 Number of queries:", t2.num_queries)
    acc2 = np.sum(t2.predict_batch(data)[0] == labels)
    print(acc1, acc2)


if __name__ == "__main__":
    test_increasing_budget_tree_iris()

import sklearn.datasets
import numpy as np

from data_structures.tree_classifier import TreeClassifier
from data_structures.wrappers.random_forest_classifier import RandomForestClassifier
import utils.utils


def test_zero_budget_tree_iris() -> None:
    iris = sklearn.datasets.load_iris()
    data, labels = iris.data, iris.target
    classes_arr = np.unique(labels)
    classes = utils.utils.class_to_idx(classes_arr)
    t = TreeClassifier(data=data, labels=labels, max_depth=5, classes=classes, budget=0)
    t.fit()
    t.tree_print()
    acc = np.sum(t.predict_batch(data)[0] == labels)
    print("MAB solution Tree Train Accuracy:", acc / len(data))


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


def test_wrapper_forest_iris() -> None:
    iris = sklearn.datasets.load_iris()
    data, labels = iris.data, iris.target
    f = RandomForestClassifier(
        data=data,
        labels=labels,
        n_estimators=20,
        max_depth=5,
    )
    f.fit()
    acc = np.sum(f.predict_batch(data)[0] == labels)
    print("Accuracy of wrapper:", (acc / len(data)))


if __name__ == "__main__":
    test_zero_budget_tree_iris()
    test_increasing_budget_tree_iris()
    test_wrapper_forest_iris()

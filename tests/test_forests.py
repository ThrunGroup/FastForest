import sklearn.datasets
from sklearn.datasets import load_diabetes
import unittest
import numpy as np
from collections import defaultdict

import ground_truth
import utils.utils
from utils import data_generator
from utils.constants import EXACT, RANDOM
from data_structures.forest_classifier import ForestClassifier
from data_structures.forest_regressor import ForestRegressor
from data_structures.tree_classifier import TreeClassifier
from data_structures.tree_regressor import TreeRegressor
from data_structures.wrappers.random_forest_classifier import RandomForestClassifier


class ForestTests(unittest.TestCase):
    # We can't have an __init__ function due to pytest providing errors about function signatures.
    np.random.seed(0)

    def test_wrapper_forest_iris(self) -> None:
        iris = sklearn.datasets.load_iris()
        data, labels = iris.data, iris.target
        f = RandomForestClassifier(
            data=data, labels=labels, n_estimators=20, max_depth=5,
        )
        f.fit()
        acc = np.sum(f.predict_batch(data)[0] == labels)
        print("Accuracy of wrapper:", (acc / len(data)))
        self.assertTrue((acc / len(data)) >= 0.97)

    def test_forest_iris(self) -> None:
        iris = sklearn.datasets.load_iris()
        data, labels = iris.data, iris.target
        f = ForestClassifier(data=data, labels=labels, n_estimators=20, max_depth=5,)
        f.fit()
        acc = np.sum(f.predict_batch(data)[0] == labels)
        print("Accuracy:", (acc / len(data)))
        self.assertTrue((acc / len(data)) >= 0.97)

    def test_ERF_iris(self) -> None:
        iris = sklearn.datasets.load_iris()
        data, labels = iris.data, iris.target
        f = ForestClassifier(
            data=data,
            labels=labels,
            n_estimators=20,
            max_depth=5,
            bin_type=RANDOM,
            num_bins=None,
        )
        f.fit()
        acc = np.sum(f.predict_batch(data)[0] == labels)
        self.assertTrue((acc / len(data)) >= 0.94)

    def test_forest_digits(self) -> None:
        digits = sklearn.datasets.load_digits()
        data, labels = digits.data, digits.target
        f = ForestClassifier(data=data, labels=labels, n_estimators=10, max_depth=5,)
        f.fit()
        acc = np.sum(f.predict_batch(data)[0] == labels)
        self.assertTrue((acc / len(data)) > 0.80)

    def test_ERF_digits(self) -> None:
        digits = sklearn.datasets.load_digits()
        data, labels = digits.data, digits.target
        f = ForestClassifier(
            data=data,
            labels=labels,
            n_estimators=10,
            max_depth=5,
            bin_type=RANDOM,
            num_bins=None,
        )
        f.fit()
        acc = np.sum(f.predict_batch(data)[0] == labels)
        self.assertTrue((acc / len(data)) > 0.84)

    def test_tree_toy(self, show: bool = False) -> None:
        toy = data_generator.create_data(10000)
        data = toy[:, :-1]
        labels = toy[:, -1]
        classes_arr = np.unique(labels)
        classes = utils.utils.class_to_idx(classes_arr)
        print("=> Ground truth:\n")
        ground_truth.ground_truth_tree(
            data, labels, show=show
        )  # Only used for interactively showing decision tree

        empty_discrete_dict = defaultdict(list)
        print("\n\n=> MAB:\n")

        # Empty discrete bins dictionary is being passed so we don't treat any features as discrete when
        # solving MAB
        print(
            "Best arm from solve_mab is: ",
            utils.solvers.solve_mab(data, labels, empty_discrete_dict),
        )

        print("\n\n=> Tree fitting:")
        t = TreeClassifier(data, labels, max_depth=3, classes=classes)
        t.fit()
        t.tree_print()

    def test_tree_iris(self) -> None:
        iris = sklearn.datasets.load_iris()
        data, labels = iris.data, iris.target
        classes_arr = np.unique(labels)
        classes = utils.utils.class_to_idx(classes_arr)
        ground_truth.ground_truth_tree(
            data=data, labels=labels, max_depth=5, show=False
        )  # Only used for interactively showing decision tree
        t = TreeClassifier(data=data, labels=labels, max_depth=5, classes=classes)
        t.fit()
        t.tree_print()
        print("Number of queries:", t.num_queries)
        acc = np.sum(t.predict_batch(data)[0] == labels)
        print("MAB solution Tree Train Accuracy:", acc / len(data))
        self.assertTrue((acc / len(data)) > 0.985)

    def test_exact_tree_iris(self) -> None:
        iris = sklearn.datasets.load_iris()
        data, labels = iris.data, iris.target
        classes_arr = np.unique(labels)
        classes = utils.utils.class_to_idx(classes_arr)

        t = TreeClassifier(
            data=data, labels=labels, max_depth=5, classes=classes, solver=EXACT
        )
        t.fit()
        t.tree_print()
        print("Number of queries:", t.num_queries)
        acc = np.sum(t.predict_batch(data)[0] == labels)
        print("MAB solution Tree Train Accuracy:", acc / len(data))
        self.assertTrue((acc / len(data)) > 0.99)

    def test_zero_budget_tree_iris(self) -> None:
        iris = sklearn.datasets.load_iris()
        data, labels = iris.data, iris.target
        classes_arr = np.unique(labels)
        classes = utils.utils.class_to_idx(classes_arr)
        t = TreeClassifier(
            data=data, labels=labels, max_depth=5, classes=classes, budget=0
        )
        t.fit()
        t.tree_print()
        acc = np.sum(t.predict_batch(data)[0] == labels)
        print("MAB solution Tree Train Accuracy:", acc / len(data))
        self.assertTrue((acc / len(data)) < 0.34)

    def test_increasing_budget_tree_iris(self) -> None:
        iris = sklearn.datasets.load_iris()
        data, labels = iris.data, iris.target
        classes_arr = np.unique(labels)
        classes = utils.utils.class_to_idx(classes_arr)

        t1 = TreeClassifier(
            data=data, labels=labels, max_depth=5, classes=classes, budget=500
        )
        t1.fit()
        t1.tree_print()
        print("T1 Number of queries:", t1.num_queries)
        acc1 = np.sum(t1.predict_batch(data)[0] == labels)
        print()
        print()
        t2 = TreeClassifier(
            data=data, labels=labels, max_depth=5, classes=classes, budget=1500
        )
        t2.fit()
        t2.tree_print()

        print("T2 Number of queries:", t2.num_queries)
        acc2 = np.sum(t2.predict_batch(data)[0] == labels)
        print(acc1, acc2)
        self.assertTrue(acc1 < acc2)


if __name__ == "__main__":
    unittest.main()

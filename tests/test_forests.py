import sklearn.datasets
import unittest
import numpy as np

from utils import data_generator
import ground_truth
from data_structures.forest_classifier import ForestClassifier
from data_structures.tree_classifier import TreeClassifier
import utils.utils


class ForestTests(unittest.TestCase):
    # We can't have an __init__ function due to pytest providing errors about function signatures.
    np.random.seed(0)

    def test_forest_iris(self) -> None:
        iris = sklearn.datasets.load_iris()
        data, labels = iris.data, iris.target
        num_classes = len(np.unique(labels))
        f = ForestClassifier(
            data=data,
            labels=labels,
            n_estimators=20,
            max_depth=5,
        )
        f.fit()
        acc = np.sum(f.predict_batch(data)[0] == labels)
        self.assertTrue((acc / len(data)) >= 0.98)

    def test_forest_digits(self) -> None:
        digits = sklearn.datasets.load_digits()
        data, labels = digits.data, digits.target
        num_classes = len(np.unique(labels))
        f = ForestClassifier(
            data=data,
            labels=labels,
            n_estimators=10,
            max_depth=5,
        )
        f.fit()
        acc = np.sum(f.predict_batch(data)[0] == labels)
        self.assertTrue((acc / len(data)) > 0.87)

    def test_tree_toy(self, show: bool = False) -> None:
        toy = data_generator.create_data(10000)
        data = toy[:, :-1]
        labels = toy[:, -1]
        classes_arr = np.unique(labels)
        classes = utils.utils.class_to_idx(classes_arr)
        print("=> Ground truth:\n")
        ground_truth.ground_truth_tree(data, labels, show=show)

        print("\n\n=> MAB:\n")
        print(
            "Best arm from solve_mab is: ", utils.mab_functions.solve_mab(data, labels)
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
        )
        t = TreeClassifier(data=data, labels=labels, max_depth=5, classes=classes)
        t.fit()
        t.tree_print()
        acc = np.sum(t.predict_batch(data)[0] == labels)
        print("MAB solution Tree Train Accuracy:", acc / len(data))

    def test_budget_tree_iris(self) -> None:
        print("Running budget test")
        iris = sklearn.datasets.load_iris()
        data, labels = iris.data, iris.target
        classes_arr = np.unique(labels)
        classes = utils.utils.class_to_idx(classes_arr)
        t = TreeClassifier(
            data=data, labels=labels, max_depth=5, classes=classes, budget=200
        )
        t.fit()
        t.tree_print()
        acc = np.sum(t.predict_batch(data)[0] == labels)
        print("MAB solution Tree Train Accuracy:", acc / len(data))


if __name__ == "__main__":
    unittest.main()

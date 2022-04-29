import sklearn.datasets
import unittest
import numpy as np

from utils import data_generator
import ground_truth
from data_structures.forest import Forest
from data_structures.tree import Tree
import utils.utils


class ForestTests(unittest.TestCase):
    # We can't have an __init__ function due to pytest providing errors about function signatures.
    np.random.seed(0)

    def test_forest_iris(self) -> None:
        iris = sklearn.datasets.load_iris()
        data, labels = iris.data, iris.target
        num_classes = len(np.unique(labels))
        f = Forest(data=data, labels=labels, n_estimators=20, max_depth=5,)
        f.fit()
        acc = np.sum(f.predict_batch(data)[0] == labels)
        self.assertTrue((acc / len(data)) >= 0.98)

    def test_forest_digits(self) -> None:
        digits = sklearn.datasets.load_digits()
        data, labels = digits.data, digits.target
        num_classes = len(np.unique(labels))
        f = Forest(data=data, labels=labels, n_estimators=10, max_depth=5,)
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

        unique_fvals_list = [np.unique(data[:, i]) for i in range(len(data[0]))]
        print("\n\n=> MAB:\n")
        print(
            "Best arm from solve_mab is: ",
            utils.mab_functions.solve_mab(data, labels, unique_fvals_list),
        )

        print("\n\n=> Tree fitting:")
        t = Tree(data, labels, max_depth=3, classes=classes)
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
        t = Tree(data=data, labels=labels, max_depth=5, classes=classes)
        t.fit()
        t.tree_print()
        acc = np.sum(t.predict_batch(data)[0] == labels)
        print("MAB solution Tree Train Accuracy:", acc / len(data))


if __name__ == "__main__":
    unittest.main()

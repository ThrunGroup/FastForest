import sklearn.datasets
import unittest
import numpy as np

from tests import ground_truth
from tests import data_generator
from data_structures.forest import Forest
from data_structures.tree import Tree


class ForestTests(unittest.TestCase):
    def __init__(self):
        np.random.seed(0)
        self.iris = sklearn.datasets.load_iris()
        self.digits = sklearn.datasets.load_digits()
        self.toy = data_generator.create_data(10000)

    def test_forest_iris(self) -> None:
        data, labels = self.iris.data, self.iris.target
        num_classes = len(np.unique(labels))
        f = Forest(
            data=data,
            labels=labels,
            n_estimators=20,
            max_depth=5,
            n_classes=num_classes,
        )
        f.fit()
        acc = np.sum(f.predict_batch(data)[0] == labels)
        self.assertTrue((acc / len(data)) > 0.99)

    def test_forest_digits(self) -> None:
        data, labels = self.digits.data, self.digits.target
        num_classes = len(np.unique(labels))
        f = Forest(
            data=data,
            labels=labels,
            n_estimators=10,
            max_depth=5,
            n_classes=num_classes,
        )
        f.fit()
        acc = np.sum(f.predict_batch(data)[0] == labels)
        self.assertTrue((acc / len(data)) > 0.87)

    def test_tree_toy(self, show: bool = False) -> None:
        data = self.toy[:, :-1]
        labels = self.toy[:, -1]
        classes_arr = np.unique(labels)
        classes = class_to_idx(classes_arr)
        print("=> Ground truth:\n")
        ground_truth.ground_truth_tree(data, labels, show=show)

        print("\n\n=> MAB:\n")
        print("Best arm from solve_mab is: ", solve_mab(data, labels))

        print("\n\n=> Tree fitting:")
        t = Tree(data, labels, max_depth=3, classes=classes)
        t.fit()
        t.tree_print()

    def test_tree_iris(self) -> None:
        iris = sklearn.datasets.load_iris()
        data, labels = iris.data, iris.target
        classes_arr = np.unique(labels)
        classes = class_to_idx(classes_arr)
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

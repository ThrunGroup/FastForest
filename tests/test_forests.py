import os

import numpy as np
import sklearn.datasets
import unittest
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from groundtruth import *
from typing import Tuple
from data_generator import create_data
from mab_functions import solve_mab
from tree import Tree
from forest import Forest


class ForestTests(unittest.TestCase):
    np.random.seed(0)
    np.set_printoptions(threshold=sys.maxsize)
    iris = sklearn.datasets.load_iris()
    digits = sklearn.datasets.load_digits()

    def test_forest_iris(self) -> None:
        self.iris = sklearn.datasets.load_iris()
        data, labels = self.iris.data, self.iris.target
        num_classes = len(np.unique(labels))
        f = Forest(
            data=data, labels=labels, n_estimators=20, max_depth=5, n_classes=num_classes
        )
        f.fit()
        acc = np.sum(f.predict_batch(data)[0] == labels)

        self.assertTrue(
            (acc / len(data)) > 0.99
        )

    def test_forest_digits(self) -> None:
        self.digits = sklearn.datasets.load_digits()
        data, labels = self.digits.data, self.digits.target
        num_classes = len(np.unique(labels))
        f = Forest(
            data=data, labels=labels, n_estimators=10, max_depth=5, n_classes=num_classes
        )
        f.fit()
        acc = np.sum(f.predict_batch(data)[0] == labels)

        print(acc/len(data))
        self.assertTrue(
            (acc / len(data)) > 0.87
        )


if __name__ == "__main__":
    unittest.main()

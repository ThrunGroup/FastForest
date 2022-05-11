import unittest
import numpy as np

from diabetes_test import test_tree_diabetes
from newsgroups_test import test_tree_news
from utils.constants import EXACT
from utils.utils import set_seed


class TestReplacement(unittest.TestCase):
    def test_tree_diabetes(self):
        seed = 1
        set_seed(seed)
        num_with, mse_with = test_tree_diabetes(seed=seed, with_replacement=True)
        num_without, mse_without = test_tree_diabetes(seed=seed, with_replacement=False)
        num_exact, mse_exact = test_tree_diabetes(seed=seed, solver=EXACT)
        self.assertTrue(
            num_without <= num_exact and abs(1 - mse_with / mse_without) < 0.05
        )

    def test_tree_news(self):
        seed = 1
        set_seed(seed)
        num_with, acc_with = test_tree_news(seed=seed, with_replacement=True)
        num_without, acc_without = test_tree_news(seed=seed, with_replacement=False)
        num_exact, acc_exact = test_tree_news(seed=seed, solver=EXACT)
        self.assertTrue(num_without <= num_exact and abs(acc_without - acc_with) < 0.02)

import unittest

from diabetes_test import test_tree_diabetes
from newsgroups_test import test_tree_news
from utils.constants import EXACT
from utils.utils import set_seed


class TestReplacement(unittest.TestCase):
    def test_tree_diabetes(self):
        seed = 3
        set_seed(seed)
        num_with, mse_with = test_tree_diabetes(
            seed=seed, with_replacement=True, print_sklearn=True
        )
        num_without, mse_without = test_tree_diabetes(seed=seed, with_replacement=False)
        num_exact, mse_exact = test_tree_diabetes(seed=seed, solver=EXACT)
        self.assertTrue(num_without <= num_exact and 1 - mse_with / mse_without < 0.04)
        print(
            f"Sample without replacement has {num_without/num_with}% of number of queries of sample with replacement "
        )
        print(
            f"Sample without replacement have {1 - mse_without/mse_with}% decrease of MSE"
        )

    def test_tree_news(self):
        seed = 0
        set_seed(seed)
        num_with, acc_with = test_tree_news(seed=seed, with_replacement=True)
        num_without, acc_without = test_tree_news(seed=seed, with_replacement=False)
        num_exact, acc_exact = test_tree_news(seed=seed, solver=EXACT)
        self.assertTrue(num_without <= num_exact and abs(acc_without - acc_with) < 0.02)
        print(
            f"Sample without replacement have {num_without / num_with}% of number of queries of sample with replacement "
        )
        print(
            f"Sample with replacement have {acc_without - acc_with}% increase of train accuracy"
        )

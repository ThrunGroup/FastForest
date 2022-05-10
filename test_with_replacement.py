import numpy as np

from test_diabetes import test_tree_diabetes, test_forest_diabetes
from test_newsgroups import test_tree_news
from utils.constants import EXACT

if __name__ == "__main__":
    seed = 1
    np.random.seed(seed)
    print("=" * 20)
    print("REGRESSION PROBLEM")
    test_tree_diabetes(seed=seed, with_replacement=True)
    test_tree_diabetes(seed=seed, with_replacement=False)
    test_tree_diabetes(seed=seed, solver=EXACT)
    print("=" * 20)
    print("Classification PROBLEM")
    test_tree_news(seed=seed, with_replacement=True)
    test_tree_news(seed=seed, with_replacement=False)
    test_tree_news(seed=seed, solver=EXACT)
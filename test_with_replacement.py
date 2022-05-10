from test_diabetes import test_tree_diabetes, test_forest_diabetes
from test_newsgroups import test_tree_news

if __name__ == "__main__":
    seed = 1
    print("=" * 20)
    print("REGRESSION PROBLEM")
    test_tree_diabetes(seed=1, with_replacement=True)
    test_tree_diabetes(seed=1, with_replacement=False)
    test_forest_diabetes(seed=1, with_replacement=True)
    test_forest_diabetes(seed=1, with_replacement=False)
    print("=" * 20)
    print("Classification PROBLEM")
    test_tree_news(seed=seed, with_replacement=True)
    test_tree_news(seed=seed, with_replacement=False)
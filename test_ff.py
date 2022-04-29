import sys
import numpy as np
from sklearn import datasets

from data_structures.tree import Tree
from utils.utils import class_to_idx


def test_tree_iris2(verbose: bool = False) -> None:
    """
    Compare tree fitting with different hyperparameters

    :param verbose: Whether to print how the trees are constructed in details.
    """
    iris = datasets.load_iris()
    data, labels = iris.data, iris.target
    classes_arr = np.unique(labels)
    classes = class_to_idx(classes_arr)

    def test_tree(max_leaf_nodes, bin_type, print_str):
        print("-" * 30)
        print(f"Fitting {print_str} tree")
        t = Tree(
            data=data,
            labels=labels,
            max_depth=5,
            classes=classes,
            max_leaf_nodes=max_leaf_nodes,
            bin_type=bin_type,
        )
        t.fit()
        if verbose:
            t.tree_print()
        acc = np.sum(t.predict_batch(data)[0] == labels)
        print("MAB solution Tree Train Accuracy:", acc / len(data))

    # Depth-first tree
    test_tree(0, "", "Depth-first splitting")

    # Best-first tree
    test_tree(32, "", "Best-first splitting")

    # Linear bin tree
    test_tree(0, "linear", "Linear bin splitting")

    # Discrete bin tree
    test_tree(0, "discrete", "Discrete bin splitting")


def main():
    print("Fitting tree iris dataset with different hyperparameters:\n")
    test_tree_iris2()


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    np.random.seed(0)
    main()

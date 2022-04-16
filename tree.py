import numpy as np
from typing import Tuple

from node import Node
from tree_classifier import TreeClassifier


class Tree(TreeClassifier):
    """
    Tree object. Contains a node attribute, the root, as well as fitting parameters that are global to the tree (i.e.,
    are used in splitting the nodes)
    """

    def __init__(
        self, data: np.ndarray, labels: np.ndarray, max_depth: int, classes: dict
    ) -> None:
        self.data = data  # TODO(@motiwari): Is this a reference or a copy?
        self.labels = labels  # TODO(@motiwari): Is this a reference or a copy?
        self.classes = classes  # a dict from class name to index
        self.idx_to_class = {value: key for key, value in classes.items()}

        self.node = Node(
            tree=self, parent=None, data=self.data, labels=self.labels, depth=0,
        )  # Root node contains all the data

        # These are copied from the link below. We won't need all of them.
        # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        self.leaves = [self.node]
        self.criterion = "GINI"
        self.splitter = "best"
        self.max_depth = 1
        self.min_samples_split = 2
        self.min_samples_leaf = 1
        self.min_weight_fraction = 0.0
        self.max_features = None
        self.random_state = None
        self.max_leaf_nodes = None
        # Make this a small negative number to avoid infinite loop when all leaves are at max_depth
        self.min_impurity_decrease = -1e-6
        self.class_weight = None
        self.ccp_alpha = 0.0

        self.depth = self.get_depth()
        self.max_depth = max_depth

    def get_depth(self) -> int:
        """
        Get the maximum depth of this tree.
        :return: an integer representing the maximum depth of any node (root = 0)
        """
        max_depth = -1
        for leaf in self.leaves:
            if leaf.depth > max_depth:
                max_depth = leaf.depth
        return max_depth

    def fit(self) -> None:
        """
        Fit the tree by recursively splitting nodes until the termination condition is reached.
        The termination condition can be a number of splits, a required reduction in impurity, or a max depth.
        Other termination conditions are to be implemented later.

        :return: None
        """
        sufficient_impurity_decrease = True
        while sufficient_impurity_decrease:
            best_leaf = None
            best_leaf_idx = None
            best_leaf_reduction = float("inf")

            # Iterate over leaves and decide which to split
            for leaf_idx, leaf in enumerate(self.leaves):

                # Do not split leaves which are already at max_depth
                if leaf.depth == self.max_depth:
                    continue

                reduction = leaf.calculate_best_split()
                if reduction:
                    reduction *= len(self.labels)
                if reduction is not None and reduction < best_leaf_reduction:
                    best_leaf = leaf
                    best_leaf_idx = leaf_idx
                    best_leaf_reduction = reduction

            if (
                best_leaf_reduction is not None
                and best_leaf_reduction < self.min_impurity_decrease
            ):
                best_leaf.split()
                split_leaf = self.leaves.pop(best_leaf_idx)
                self.leaves.append(split_leaf.left)
                self.leaves.append(split_leaf.right)
            else:
                sufficient_impurity_decrease = False

            self.depth = self.get_depth()

        print("Fitting finished")

    def predict(self, datapoint: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Calculate the predicted probabilities that the given datapoint belongs to each classifier

        :param datapoint: datapoint to fit
        :return: the probabilities of the datapoint being each class label
        """
        node = self.node
        while node.left:
            feature_value = datapoint[node.split_on]
            if feature_value <= node.split_value:
                node = node.left
            else:
                node = node.right
        assert node.right is None, "Tree is malformed"

        probs = node.counts / np.sum(node.counts)
        label_pred = self.idx_to_class[
            probs.argmax()
        ]  # Find ith key of dictionary
        assert np.allclose(probs.sum(), 1), "Probabilities don't sum to 1"
        return label_pred, probs

    def tree_print(self) -> None:
        """
        Print the tree depth-first in a format matching sklearn
        """
        self.node.n_print()
        print("\n")  # For consistency with sklearn

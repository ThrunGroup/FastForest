import numpy as np
from typing import Tuple, List

from data_structures.node import Node
from data_structures.tree_classifier import TreeClassifier


class Tree(TreeClassifier):
    """
    Tree object. Contains a node attribute, the root, as well as fitting parameters that are global to the tree (i.e.,
    are used in splitting the nodes)
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        max_depth: int,
        classes: dict,
        min_samples_split: int = 2,
        min_impurity_decrase: float = -1e-6,
        max_leaf_nodes: int = 0,
        features_list: List[np.ndarray] = None,
        bin_type: str = "linear"
    ) -> None:
        self.data = data  # TODO(@motiwari): Is this a reference or a copy?
        self.labels = labels  # TODO(@motiwari): Is this a reference or a copy?
        self.n_data = len(labels)
        self.classes = classes  # dict from class name to class index
        self.idx_to_class = {value: key for key, value in classes.items()}
        self.bin_type = bin_type

        self.node = Node(
            tree=self, parent=None, data=self.data, labels=self.labels, depth=0, bin_type=self.bin_type
        )  # Root node contains all the data

        # These are copied from the link below. We won't need all of them.
        # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        self.leaves = []
        self.criterion = "GINI"
        self.splitter = "best"
        self.max_depth = 1
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = 1
        self.min_weight_fraction = 0.0
        self.max_features = None
        self.random_state = None
        self.max_leaf_nodes = max_leaf_nodes
        # Make this a small negative number to avoid infinite loop when all leaves are at max_depth
        self.min_impurity_decrease = min_impurity_decrase
        self.class_weight = None
        self.ccp_alpha = 0.0
        self.depth = 1
        self.max_depth = max_depth

        if not features_list:
            self.features_list = [np.unique(data[:, i]) for i in range(len(data[0]))]
        else:
            self.features_list = features_list

    def get_depth(self) -> int:
        """
        Get the maximum depth of this tree.
        :return: an integer representing the maximum depth of any node (root = 0)
        """
        max_depth = -1
        return max([leaf.depth for leaf in self.leaves])

    def is_terminate(self, node: Node) -> bool:
        """
        Check whether the node satisfies the termination condition of splitting.

        :param node: A node which is considered
        :return: Whether to terminate splitting a node
        """
        node.is_check_terminate = True
        return (
            self.max_depth <= node.depth
            or self.min_samples_split >= node.n_data
            or self.min_impurity_decrease
            <= node.calculate_best_split() * node.n_data / self.n_data
        )

    def fit(self, verbose=True) -> None:
        """
        Fit the tree by recursively splitting nodes until the termination condition is reached.
        The termination condition can be a number of splits, a required reduction in impurity, or a max depth.
        Other termination conditions are to be implemented later.

        :return: None
        """
        if self.max_leaf_nodes:  # Best-first tree fitting
            self.leaves.append(self.node)  # Append root node to self.leaves
            while len(self.leaves) < self.max_leaf_nodes:
                all_terminate = True
                best_leaf = None
                best_leaf_idx = None
                best_leaf_reduction = float("inf")

                # Iterate over leaves and decide which to split
                for leaf_idx, leaf in enumerate(self.leaves):
                    reduction = (
                        leaf.calculate_best_split() * leaf.n_data / self.n_data
                    )  # Weighted impurity reduction
                    if not leaf.is_check_terminate:
                        leaf.is_terminate = self.is_terminate(leaf)
                    if not leaf.is_terminate:
                        if reduction <= best_leaf_reduction:
                            best_leaf = leaf
                            best_leaf_idx = leaf_idx
                            best_leaf_reduction = reduction
                if best_leaf is None:  # All the nodes satisfy the termination condition
                    break
                best_leaf.split()
                split_leaf = self.leaves.pop(best_leaf_idx)
                split_leaf.prediction_probs = None  # this node is no longer a leaf
                self.leaves.append(split_leaf.left)
                self.leaves.append(split_leaf.right)
                self.depth = self.get_depth()

        else:  # Depth-first tree fitting
            self.recursive_split(self.node)

        if verbose:
            print("Fitting finished")

    def recursive_split(self, node: Node) -> None:
        """
        Recursively split nodes till the termination condition is satisfied

        :param node: A root node to be split recursively
        """
        node.is_terminate = self.is_terminate(node)
        if node.is_terminate:
            self.leaves.append(node)
            return
        else:
            node.calculate_best_split()
            node.split()
            self.recursive_split(node.left)
            self.recursive_split(node.right)

    def predict(self, datapoint: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Calculate the predicted probabilities that the given datapoint belongs to each classifier

        :param datapoint: datapoint to fit
        :return: the probabilities of the datapoint being each class label
        """
        node = self.node
        while node.left:
            feature_value = datapoint[node.split_feature]
            node = node.left if feature_value <= node.split_value else node.right
        assert node.right is None, "Tree is malformed"

        # The prediction probability has been cached
        if node.prediction_probs is not None:
            return node.predicted_label, node.prediction_probs

        # otherwise, make prediction and cache it
        node.prediction_probs = node.counts / np.sum(node.counts)
        node.predicted_label = self.idx_to_class[
            node.prediction_probs.argmax()
        ]  # Find ith key of dictionary
        assert np.allclose(
            node.prediction_probs.sum(), 1
        ), "Probabilities don't sum to 1"
        return node.predicted_label, node.prediction_probs

    def tree_print(self) -> None:
        """
        Print the tree depth-first in a format matching sklearn
        """
        self.node.n_print()
        print("\n")  # For consistency with sklearn

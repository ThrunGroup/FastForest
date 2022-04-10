import numpy as np

from typing import List

from node import Node

class Tree:
    """
    Tree object. Contains a node attribute, the root, as well as fitting parameters that are global to the tree (i.e.,
    are used in splitting the nodes)
    """
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = data  # TODO(@motiwari): Is this a reference or a copy?
        self.labels = labels  # TODO(@motiwari): Is this a reference or a copy?
        self.node = Node(tree=self, data=self.data, labels=self.labels, depth=0)  # Root node contains all the data
        self.n_classes = 2

        # These are copied from the link below. We won't need all of them.
        # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        self.leaves = [self.node]
        self.criterion = 'gini'
        self.splitter = 'best'
        self.max_depth = None
        self.min_samples_split = 2
        self.min_samples_leaf = 1
        self.min_weight_fraction = 0.0
        self.max_features = None
        self.random_state = None
        self.max_leaf_nodes = None
        self.min_impurity_decrease = 0.0
        self.class_weight = None
        self.ccp_alpha = 0.0


    def fit(self) -> None:
        sufficient_impurity_decrease = True
        while sufficient_impurity_decrease:
            best_leaf = None
            best_leaf_idx = None
            best_leaf_reduction = None

            # Iterate over leaves and decide which to split
            for leaf_idx, leaf in enumerate(self.leaves):
                reduction = leaf.calculate_best_split()
                if reduction < best_leaf_reduction:
                    best_leaf = leaf
                    best_leaf_idx = leaf_idx
                    best_leaf_reduction = reduction

            if best_leaf_reduction < self.min_impurity_decrease:
                best_leaf.split()
                split_leaf = self.leaves.pop(best_leaf_idx)
                self.leaves.append(split_leaf.left)
                self.leaves.append(split_leaf.right)
            else:
                sufficient_impurity_decrease = False

        print("Fitting finished")

    def tree_print(self) -> None:
        """
        Print the tree depth-first
        """
        self.node.n_print()











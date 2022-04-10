from __future__ import annotations

import numpy as np
from fast_forest import solve_mab

# We need to do this below to avoid the circular import: Tree <--> Node
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tree import Tree

class Node:
    def __init__(self, tree: Tree, data: np.ndarray, labels: np.ndarray, depth: int) -> None:
        self.tree = tree
        self.data = data  # TODO(@motiwari): Is this a reference or a copy?
        self.labels = labels
        self.depth = depth
        self.left = None
        self.right = None

        self.zeros = len(np.where(labels == 0)[0])
        self.ones = len(np.where(labels == 1)[0])

        self.split_on = None
        self.split_feature = None
        self.split_value = None
        self.split_reduction = None


    def calculate_best_split(self):
        """
        Speculatively calculate the best split
        :return: None, but assign
        """
        # Use MAB solution here
        self.split_feature, self.split_value, self.split_reduction = solve_mab(self.data, self.labels)


    def split(self) -> None:
        """
        Splits the node into two children nodes.
        :return: None
        """
        if self.split_on is not None:
            raise Exception('Error: this node is already split')

        if self.split_feature is None:
            self.calculate_best_split()

        assert self.split_reduction < 0, "Error: splitting this node would increase impurity. Should never be here"

        # Creat left and right children with appropriate datasets
        # NOTE: Asymmetry with <= and >
        left_data = self.data[np.where(self.data[self.split_feature <= self.split_value])]
        left_labels = self.labels[np.where(self.data[self.split_feature <= self.split_value])]
        self.left = Node(self.tree, left_data, left_labels, self.depth + 1)

        right_data = self.data[np.where(self.data[self.split_feature > self.split_value])]
        right_labels = self.labels[np.where(self.data[self.split_feature > self.split_value])]
        self.right = Node(self.tree, right_data, right_labels, self.depth + 1)

        self.split_on = self.split_feature


    def n_print(self) -> None:
        """
        Print the node's children depth-first
        Me: split x < 5:

        """
        print('\t' * self.depth,
              "Split on feature: ", self.split_feature,
              " at ", self.split_value,
              " for split reduction ", self.split_reduction,
              )

        assert (self.left and self.right) or (self.left is None and self.right is None), "Error: split is malformed"
        if self.left:
            self.left.print()
            self.right.print()
        else:
            print('\t' * self.depth, "Zeros:", self.zeros, ", Ones:", self.ones)



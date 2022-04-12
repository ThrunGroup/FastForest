from __future__ import annotations

import numpy as np
from fast_forest import solve_mab

# We need to do this below to avoid the circular import: Tree <--> Node
# See https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tree import Tree


class Node:
    def __init__(
        self, tree: Tree, parent: Node, data: np.ndarray, labels: np.ndarray, depth: int, num_classes: int = 2
    ) -> None:
        self.tree = tree
        self.parent = parent  # To allow walking back upwards
        self.data = data  # TODO(@motiwari): Is this a reference or a copy?
        self.labels = labels
        self.depth = depth
        self.left = None
        self.right = None

        # Assume labels are all integers from 0 to i.
        self.num_classes = num_classes
        self.counts = np.zeros(self.num_classes, dtype=int)
        for class_ in labels:
            self.counts[int(class_)] += 1

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
        self.split_feature, self.split_value, self.split_reduction = solve_mab(
            self.data, self.labels
        )
        return self.split_reduction

    def split(self) -> None:
        """
        Splits the node into two children nodes.
        :return: None
        """
        if self.split_on is not None:
            raise Exception("Error: this node is already split")

        if self.split_feature is None:
            _ = self.calculate_best_split()

        # Verify that splitting would actually help
        if self.split_reduction is not None:
            assert (
                self.split_reduction < 0
            ), "Error: splitting this node would increase impurity. Should never be here"

        # Creat left and right children with appropriate datasets
        # NOTE: Asymmetry with <= and >
        left_idcs = np.where(self.data[:, self.split_feature] <= self.split_value)
        left_data = self.data[left_idcs]
        left_labels = self.labels[left_idcs]
        self.left = Node(self.tree, left_data, left_labels, self.depth + 1)

        right_idcs = np.where(self.data[:, self.split_feature] > self.split_value)
        right_data = self.data[right_idcs]
        right_labels = self.labels[right_idcs]
        self.right = Node(self.tree, right_data, right_labels, self.depth + 1)

        self.split_on = self.split_feature

    def n_print(self) -> None:
        """
        Print the node's children depth-first
        Me: split x < 5:

        """
        assert (self.left and self.right) or (
            self.left is None and self.right is None
        ), "Error: split is malformed"
        if self.left:
            print(
                ("|   " * self.depth)
                + "|--- feature_"
                + str(self.split_on)
                + " <= "
                + str(self.split_value)
            )
            self.left.n_print()
            print(
                ("|   " * self.depth)
                + "|--- feature_"
                + str(self.split_on)
                + " > "
                + str(self.split_value)
            )
            self.right.n_print()
        else:
            class_predicted = np.argmax(self.counts)
            print(("|   " * self.depth) + "|--- " + "class: " + str(class_predicted))

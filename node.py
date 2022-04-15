from __future__ import annotations

import numpy as np
from mab_functions import solve_mab

from utils import type_check

type_check()


class Node:
    def __init__(
        self,
        tree: Tree,
        parent: Node,
        data: np.ndarray,
        labels: np.ndarray,
        depth: int,
    ) -> None:
        self.tree = tree
        self.parent = parent  # To allow walking back upwards
        self.data = data  # TODO(@motiwari): Is this a reference or a copy?
        self.labels = labels
        self.depth = depth
        self.left = None
        self.right = None

        # NOTE: Not assume labels are all integers from 0 to num_classes-1
        existing_classes = np.unique(
            labels
        )  # Note: It seems it's not efficient, but actually it is since it reduces
        # the computation of finding counts of labels which aren't in "self.labels"
        self.counts = np.zeros(
            len(self.tree.classes)
        )  # self.tree classes contains all classes of original data
        for class_ in existing_classes:
            class_idx = self.tree.classes[class_]
            self.counts[class_idx] += len(np.where(labels == class_)[0])

        self.split_on = None
        self.split_feature = None
        self.split_value = None
        self.split_reduction = None
        self.is_calculate_best = False

    def calculate_best_split(self):
        """
        Speculatively calculate the best split
        :return: None, but assign
        """
        if self.is_calculate_best:
            return (
                self.split_reduction
            )  # If we already calculate it, return self.split_reduction right away
        self.is_calculate_best = True
        results = solve_mab(self.data, self.labels)
        if results is not None:
            self.split_feature, self.split_value, self.split_reduction = results
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
            self.left = Node(self.tree, self, left_data, left_labels, self.depth + 1,)

            right_idcs = np.where(self.data[:, self.split_feature] > self.split_value)
            right_data = self.data[right_idcs]
            right_labels = self.labels[right_idcs]
            self.right = Node(
                self.tree, self, right_data, right_labels, self.depth + 1,
            )

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
            class_idx_pred = np.argmax(self.counts)
            class_pred = list(self.tree.classes.keys())[class_idx_pred] # print class name not class index
            print(("|   " * self.depth) + "|--- " + "class: " + str(class_pred))

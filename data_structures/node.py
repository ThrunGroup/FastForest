from __future__ import (
    annotations,
)  # For typechecking parent: Node, this is somehow important

import numpy as np

from utils.mab_functions import solve_mab
from utils.utils import type_check, counts_of_labels

type_check()


class Node:
    def __init__(
        self,
        tree: Tree,
        parent: Node,
        data: np.ndarray,
        labels: np.ndarray,
        depth: int,
        proportion: float,
    ) -> None:
        self.tree = tree
        self.parent = parent  # To allow walking back upwards
        self.data = data  # TODO(@motiwari): Is this a reference or a copy?
        self.labels = labels
        self.depth = depth
        self.proportion = proportion
        self.left = None
        self.right = None

        # NOTE: Do not assume labels are all integers from 0 to num_classes-1
        self.counts = counts_of_labels(self.tree.classes, labels)

        # We need a separate variable for already_split, because self.split_feature can be truthy
        # even if the split hasn't been performed
        self.already_split = False
        self.split_feature = None
        self.split_value = None

        # values to cache
        self.best_reduction_computed = False
        self.num_queries = 0
        self.split_reduction = None
        self.prediction_probs = None
        self.predicted_label = None

    def calculate_best_split(self) -> None:
        """
        Speculatively calculate the best split

        :return: None, but assign
        """
        if self.best_reduction_computed:
            return self.split_reduction  # If we already calculated it, return it

        results = solve_mab(self.data, self.labels)
        # Even if results is None, we should cache the fact that we know that
        self.best_reduction_computed = True

        if results is not None:
            (
                self.split_feature,
                self.split_value,
                self.split_reduction,
                self.num_queries,
            ) = results
            self.split_reduction *= self.proportion  # Normalize by number of datapoints
            return self.split_reduction

    def create_child_node(self, idcs: np.ndarray) -> Node:
        child_data = self.data[idcs]
        child_labels = self.labels[idcs]
        return Node(
            self.tree,
            self,
            child_data,
            child_labels,
            self.depth + 1,
            self.proportion * (len(child_labels) / len(self.labels)),
        )

    def split(self) -> None:
        """
        Splits the node into two children nodes.

        :return: None
        """
        if self.already_split:
            raise Exception("Error: this node is already split")

        if self.split_feature is None:
            _ = self.calculate_best_split()

        # Verify that splitting would actually help
        if self.split_reduction is not None:
            assert (
                self.split_reduction < 0
            ), "Error: splitting this node would increase impurity. Should never be here"

            # NOTE: Asymmetry with <= and >
            left_idcs = np.where(self.data[:, self.split_feature] <= self.split_value)
            self.left = self.create_child_node(left_idcs)

            right_idcs = np.where(self.data[:, self.split_feature] > self.split_value)
            self.right = self.create_child_node(right_idcs)

            # Reset cached prediction values
            self.prediction_probs = None
            self.predicted_label = None

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
                + str(self.already_split)
                + " <= "
                + str(self.split_value)
            )
            self.left.n_print()
            print(
                ("|   " * self.depth)
                + "|--- feature_"
                + str(self.already_split)
                + " > "
                + str(self.split_value)
            )
            self.right.n_print()
        else:
            class_idx_pred = np.argmax(self.counts)
            class_pred = self.tree.idx_to_class[
                class_idx_pred
            ]  # print class name not class index
            print(("|   " * self.depth) + "|--- " + "class: " + str(class_pred))

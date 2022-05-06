from __future__ import (
    annotations,
)  # For typechecking parent: Node, this is somehow important
from typing import Dict, Union
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
        is_classification: bool = True,
        verbose: bool = True,
        bin_type: str = "",
    ) -> None:
        self.tree = tree
        self.parent = parent  # To allow walking back upwards
        self.data = data  # TODO(@motiwari): Is this a reference or a copy?
        self.labels = labels
        self.n_data = len(labels)
        self.bin_type = bin_type
        self.depth = depth
        self.proportion = proportion
        self.is_classification = is_classification
        self.left = None
        self.right = None
        self.verbose = verbose

        # NOTE: Do not assume labels are all integers from 0 to num_classes-1
        if is_classification:
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
        self.is_splittable = None
        if self.is_classification:
            self.prediction_probs = None
            self.predicted_label = None
        else:
            self.predicted_value = None

    def calculate_best_split(self) -> Union[float, int]:
        """
        Speculatively calculate the best split

        :return: Weighted impurity reduction of the node's best split
        """
        if self.best_reduction_computed:
            return self.split_reduction

        results = solve_mab(
            self.data,
            self.labels,
            self.tree.discrete_features,
            fixed_bin_type=self.bin_type,
            is_classification=self.is_classification,
        )
        # Even if results is None, we should cache the fact that we know that
        self.best_reduction_computed = True

        if type(results) == tuple:  # Found a solution
            (
                self.split_feature,
                self.split_value,
                self.split_reduction,
                self.num_queries,
            ) = results
            self.split_reduction *= self.proportion  # Normalize by number of datapoints
            if self.verbose:
                print("Calculated split with", self.num_queries, "queries")
            return self.split_reduction
        else:
            self.num_queries = results
            if self.verbose:
                print("Calculated split with", self.num_queries, "queries")
            self.num_queries = results

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
            bin_type=self.bin_type,
            is_classification=self.is_classification,
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
            self.predicted_value = None
            self.already_split = True

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
                + str(self.split_feature)
                + " <= "
                + str(self.split_value)
            )
            self.left.n_print()
            print(
                ("|   " * self.depth)
                + "|--- feature_"
                + str(self.split_feature)
                + " > "
                + str(self.split_value)
            )
            self.right.n_print()
        else:
            if self.is_classification:
                class_idx_pred = np.argmax(self.counts)
                class_pred = self.tree.idx_to_class[
                    class_idx_pred
                ]  # print class name not class index
                print(("|   " * self.depth) + "|--- " + "class: " + str(class_pred))
            else:
                print(
                    ("|   " * self.depth) + "|--- " + "value: ",
                    float(np.mean(self.labels)),
                )

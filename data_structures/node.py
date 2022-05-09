from __future__ import (
    annotations,
)  # For typechecking parent: Node, this is somehow important
import numpy as np
import math
from typing import Union
from collections import defaultdict

from utils.solvers import solve_mab, solve_exactly
from utils.utils import type_check, counts_of_labels
from utils.constants import MAB, EXACT, GINI, LINEAR, SQRT

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
        bin_type: str = LINEAR,
        criterion: str = GINI,
        solver: str = MAB,
        verbose: bool = True,
        erf_k: str = "",
        feature_subsampling: Union[str, int] = None,
    ) -> None:
        self.tree = tree
        self.parent = parent  # To allow walking back upwards
        self.data = data  # TODO(@motiwari): Is this a reference or a copy?
        self.labels = labels
        self.n_data = len(labels)
        self.bin_type = bin_type
        self.erf_k = erf_k
        self.depth = depth
        self.proportion = proportion
        self.is_classification = is_classification
        self.left = None
        self.right = None
        self.verbose = verbose
        self.solver = solver
        self.criterion = criterion
        self.feature_subsampling = feature_subsampling
        self.discrete_features = defaultdict(list)

        # Subsample features
        N = len(self.data[0])
        if feature_subsampling is None:
            self.feature_idcs = np.arange(N)
        elif feature_subsampling == SQRT:
            self.feature_idcs = np.random.choice(N, math.ceil(math.sqrt(N)), replace=False)
        elif type(feature_subsampling) == int:  # If int, subsample feature_subsampling(int) features.
            self.feature_idcs = np.random.choice(N, feature_subsampling, replace=False)
        else:
            raise NotImplementedError("Invalid type of feature_subsampling")

        # New discrete_features corresponding to new feature indices
        i = 0
        for feature_idx in self.feature_idcs:
            self.discrete_features[i] = self.tree.discrete_features[feature_idx]
            i += 1

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

        if self.solver == MAB:
            results = solve_mab(
                data=self.data[:, self.feature_idcs],
                labels=self.labels,
                discrete_bins_dict=self.discrete_features,
                fixed_bin_type=self.bin_type,
                is_classification=self.is_classification,
                impurity_measure=self.criterion
            )
        elif self.solver == EXACT:
            results = solve_exactly(
                data=self.data[:, self.feature_idcs],
                labels=self.labels,
                discrete_bins_dict=self.discrete_features,
                fixed_bin_type=self.bin_type,
                is_classification=self.is_classification,
                impurity_measure=self.criterion
            )
        else:
            raise Exception("Invalid solver specified, must be MAB or EXACT")

        # Even if results is None, we should cache the fact that we know that
        self.best_reduction_computed = True

        if type(results) == tuple:  # Found a solution
            (
                self.split_feature,
                self.split_value,
                self.split_reduction,
                self.num_queries,
            ) = results
            self.split_feature = self.feature_idcs[self.split_feature]  # Feature index of original dataset
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
            tree=self.tree,
            parent=self,
            data=child_data,
            labels=child_labels,
            depth=self.depth + 1,
            proportion=self.proportion * (len(child_labels) / len(self.labels)),
            bin_type=self.bin_type,
            is_classification=self.is_classification,
            solver=self.solver,
            verbose=self.verbose,
            criterion=self.criterion,
            feature_subsampling=self.feature_subsampling
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

from __future__ import (
    annotations,
)  # For typechecking parent: Node, this is somehow important
import numpy as np
from numba import jit
from typing import Union
from collections import defaultdict
from utils.utils import get_subset_2d

from utils.solvers import solve_mab, solve_exactly, solve_randomly
from utils.utils import (
    type_check,
    counts_of_labels,
    choose_features,
    remap_discrete_features,
)
from utils.constants import (
    MAB,
    EXACT,
    RANDOM_SOLVER,
    GINI,
    LINEAR,
    DEFAULT_NUM_BINS,
    BATCH_SIZE,
)

type_check()


class Node:
    def __init__(
        self,
        tree: Tree,
        parent: Node,
        idcs: np.ndarray,
        depth: int,
        proportion: float,
        is_classification: bool = True,
        bin_type: str = LINEAR,
        num_bins: int = DEFAULT_NUM_BINS,
        criterion: str = GINI,
        solver: str = MAB,
        verbose: bool = True,
        feature_subsampling: Union[str, int] = None,
        with_replacement: bool = False,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self.tree = tree
        self.feature_subsampling = feature_subsampling
        # To decrease memory usage and cost for making a copy of large array, we don't pass data array to child node
        # but indices
        # The features aren't global to the tree, so we should be resampling the features at every node

        # TODO(@motiwari): Change "None" back to self.feature_subsampling
        self.feature_idcs = choose_features(
            self.tree.feature_idcs, None, self.tree.rng
        )

        if self.tree.discrete_features is not None:
            self.discrete_features = remap_discrete_features(
                self.feature_idcs, self.tree.discrete_features
            )
        self.idcs = idcs
        self.parent = parent  # To allow walking back upward
        self.data = get_subset_2d(self.tree.data, self.idcs, self.feature_idcs)
        self.labels = self.tree.labels[idcs]
        self.n_data = len(self.labels)
        self.bin_type = bin_type
        self.num_bins = num_bins
        self.depth = depth
        self.proportion = proportion
        self.is_classification = is_classification
        self.left = None
        self.right = None
        self.verbose = verbose
        self.solver = solver
        self.criterion = criterion
        self.with_replacement = with_replacement
        self.discrete_features = self.tree.discrete_features
        self.batch_size = batch_size

        # Reindex minmax
        if self.tree.minmax is not None:
            self.minmax = (
                self.tree.minmax[0][self.feature_idcs],
                self.tree.minmax[1][self.feature_idcs],
            )
        else:
            self.minmax = None

        # NOTE: Do not assume labels are all integers from 0 to num_classes-1
        if is_classification:
            self.counts = counts_of_labels(self.tree.classes, self.labels)

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

    def calculate_best_split(self, budget: int = None) -> Union[float, int]:
        """
        Speculatively calculate the best split

        :return: Weighted impurity reduction of the node's best split
        """
        if self.best_reduction_computed:
            return self.split_reduction

        if self.tree.use_logarithmic_split:
            self.num_bins = int(np.log2(self.n_data)) + 1
        if self.tree.use_dynamic_epsilon:
            self.epsilon = self.tree.epsilon * np.sqrt(self.depth)
        else:
            self.epsilon = self.tree.epsilon

        if self.solver == MAB:
            results = solve_mab(
                data=self.data,
                labels=self.labels,
                minmax=self.minmax,
                discrete_bins_dict=self.discrete_features,
                binning_type=self.bin_type,
                num_bins=self.num_bins,
                is_classification=self.is_classification,
                impurity_measure=self.criterion,
                with_replacement=self.with_replacement,
                budget=budget,
                epsilon=self.epsilon,
                batch_size=self.batch_size,
                rng=self.tree.rng,
            )
        elif self.solver == EXACT:
            results = solve_exactly(
                data=self.data,
                labels=self.labels,
                minmax=self.minmax,
                discrete_bins_dict=self.discrete_features,
                binning_type=self.bin_type,
                num_bins=self.num_bins,
                is_classification=self.is_classification,
                impurity_measure=self.criterion,
                # NOTE: not implemented with budget yet
            )
        elif self.solver == RANDOM_SOLVER:
            results = solve_randomly(
                data=self.data,
                labels=self.labels,
                minmax=self.minmax,
                discrete_bins_dict=self.discrete_features,
                binning_type=self.bin_type,
                num_bins=self.num_bins,
                is_classification=self.is_classification,
                impurity_measure=self.criterion,
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
            self.prev_split_feature = self.split_feature
            self.split_feature = self.feature_idcs[
                self.split_feature
            ]  # Feature index of original datasets
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
        return Node(
            tree=self.tree,
            parent=self,
            idcs=idcs,
            depth=self.depth + 1,
            proportion=self.proportion * (len(idcs) / len(self.labels)),
            bin_type=self.bin_type,
            num_bins=self.num_bins,
            is_classification=self.is_classification,
            solver=self.solver,
            verbose=self.verbose,
            criterion=self.criterion,
            feature_subsampling=self.feature_subsampling,
            with_replacement=self.with_replacement,
            batch_size=self.batch_size,
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
            left_idcs = self.idcs[
                np.where(self.data[:, self.prev_split_feature] <= self.split_value)
            ]
            right_idcs = self.idcs[
                np.where(self.data[:, self.prev_split_feature] > self.split_value)
            ]

            if len(left_idcs) == 0 or len(right_idcs) == 0:
                # Our MAB erroneously identified a split as good, but it actually wasn't and puts all the children on
                # one side
                self.already_split = True
                raise AttributeError("Wrong split!!")
            else:
                self.left = self.create_child_node(left_idcs)
                self.right = self.create_child_node(right_idcs)

                # Reset cached prediction values
                self.prediction_probs = None
                self.predicted_label = None
                self.predicted_value = None
                self.already_split = True

                # Free memory
                del self.data, self.labels

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

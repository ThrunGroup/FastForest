import numpy as np
import math

from collections import defaultdict
from abc import ABC
from typing import Union, Tuple, DefaultDict

from data_structures.node import Node
from utils.utils import (
    data_to_discrete,
    set_seed,
    choose_features,
    remap_discrete_features,
)
from utils.constants import (
    MAB,
    EXACT,
    LINEAR,
    IDENTITY,
    BEST,
    DEPTH,
    GINI,
    DEFAULT_NUM_BINS,
    DEFAULT_MIN_IMPURITY_DECREASE,
    BATCH_SIZE,
)


class TreeBase(ABC):
    """
    TreeBase class. Contains a node attribute, the root, as well as fitting parameters that are global to the tree (i.e.,
    are used in splitting the nodes). TreeClassifier and TreeRegressor will inherit TreeBase class.
    """

    def __init__(
        self,
        data: np.ndarray = None,
        labels: np.ndarray = None,
        max_depth: int = 100,
        classes: dict = None,
        feature_subsampling: Union[str, int] = None,
        tree_global_feature_subsampling: bool = False,
        min_samples_split: int = 2,
        min_impurity_decrease: float = DEFAULT_MIN_IMPURITY_DECREASE,
        max_leaf_nodes: int = None,
        discrete_features: DefaultDict = None,
        bin_type: str = LINEAR,
        num_bins: int = DEFAULT_NUM_BINS,
        budget: int = None,
        is_classification: bool = True,
        criterion: str = GINI,
        splitter: str = BEST,
        solver: str = MAB,
        random_state: int = 0,
        with_replacement: bool = False,
        verbose: bool = False,
        make_discrete: bool = False,
        minmax: Tuple[np.ndarray, np.ndarray] = None,
        use_logarithmic_split: bool = False,
        use_dynamic_epsilon: bool = False,
        epsilon: float = 0,
        batch_size: int = BATCH_SIZE,
        idcs: np.ndarray = None,
    ) -> None:
        self.data = data  # This is a REFERENCE
        self.labels = labels  # This is a REFERENCE
        if self.labels is not None:
            self.n_data = len(labels)
        assert len(self.data) == len(
            self.labels
        ), "Data and labels must have the same size"
        self.max_depth = max_depth
        self.minmax = (
            minmax  # minmax = (minimum array of features, maximum array of features)
        )
        if is_classification:
            self.classes = classes  # dict from class name to class index
            self.idx_to_class = {value: key for key, value in classes.items()}

        self.feature_subsampling = feature_subsampling
        self.tree_global_feature_subsampling = tree_global_feature_subsampling
        self.discrete_features = discrete_features
        self.make_discrete = make_discrete
        if (bin_type == LINEAR) or (bin_type == IDENTITY):
            self.make_discrete = False

        self.min_samples_split = min_samples_split
        # Make this a small negative number to avoid infinite loop when all leaves are at max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.max_leaf_nodes = max_leaf_nodes

        self.bin_type = bin_type
        self.num_bins = num_bins

        self.remaining_budget = budget
        self.is_classification = is_classification
        self.criterion = criterion
        self.splitter = splitter
        self.solver = solver
        self.random_state = random_state
        set_seed(self.random_state)
        self.rng = np.random.default_rng(random_state)
        self.with_replacement = with_replacement
        self.verbose = verbose
        self.use_logarithmic_split = use_logarithmic_split
        self.use_dynamic_epsilon = use_dynamic_epsilon
        self.epsilon = epsilon

        if idcs is None:
            idcs = np.arange(self.n_data)
        self.node = Node(
            tree=self,
            parent=None,
            idcs=idcs,  # Root node contains all the data
            depth=0,
            proportion=1.0,
            bin_type=self.bin_type,
            num_bins=self.num_bins,
            is_classification=self.is_classification,
            verbose=self.verbose,
            solver=self.solver,
            criterion=self.criterion,
            feature_subsampling=self.feature_subsampling,
            tree_global_feature_subsampling=self.tree_global_feature_subsampling,
            with_replacement=self.with_replacement,
            batch_size=batch_size,
        )

        # These are copied from the link below. We won't need all of them.
        # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        self.leaves = []

        self.min_samples_leaf = 1
        self.min_weight_fraction = 0.0
        self.max_features = None
        self.class_weight = None
        self.ccp_alpha = 0.0
        self.depth = 1

        self.num_splits = 0
        self.num_queries = 0

    def get_depth(self) -> int:
        """
        Get the maximum depth of this tree.
        :return: an integer representing the maximum depth of any node (root = 0)
        """
        return max([leaf.depth for leaf in self.leaves])

    def check_splittable_constraints(self, node: Node) -> bool:
        """
        Check whether the node satisfies the splittable condition of splitting that "do not" call
        node.calculate_best_split().

        :param node: A node which is considered
        :return: Whether it's possible to split a node
        """

        return (
            self.max_depth > node.depth
            and self.min_samples_split < node.n_data
            and len(np.unique(node.labels)) > 1
        )

    def check_splittable_impurity(self, node: Node) -> bool:
        """
        Check whether the node satisfies the splittable condition of splitting that does call
        node.calculate_best_split().

        :param node: A node which is considered
        :return: Whether it's possible to split a node
        """

        return (
            node.calculate_best_split() is not None
            and self.min_impurity_decrease
            > node.calculate_best_split() * node.n_data / self.n_data
        )

    def check_splittable(self, node: Node) -> bool:
        """
        Check whether the node satisfies the splittable condition of splitting.
        Note: incurs a call to node.calculate_best_split()

        :param node: A node which is considered
        :return: Whether it's possible to split a node
        """
        return self.check_splittable_constraints(
            node
        ) and self.check_splittable_impurity(node)

    @staticmethod
    def check_both_or_neither(
        data: np.ndarray = None, labels: np.ndarray = None
    ) -> bool:
        if data is None:
            if labels is not None:
                raise Exception("Need to pass both data and labels to .fit()")
        else:
            if labels is None:
                raise Exception("Need to pass both data and labels to .fit()")

        # Either (data and labels) or (not data and not labels)
        return True

    def fit(self, data: np.ndarray = None, labels: np.ndarray = None) -> None:
        """
        Fit the tree by recursively splitting nodes until the termination condition is reached.
        The termination condition can be a number of splits, a required reduction in impurity, or a max depth.
        Other termination conditions are to be implemented later.

        :return: None
        """
        # Imitate the structure of fit method of sklearn
        self.check_both_or_neither(data, labels)
        if data is not None:
            self.data = data
            self.labels = labels
            self.n_data = len(labels)

        if self.make_discrete:
            self.discrete_features = data_to_discrete(self.data, n=10)

        if self.tree_global_feature_subsampling:
            # Sample the features randomly once, to be used in the entire tree
            self.feature_idcs = choose_features(self.data, self.feature_subsampling)
            self.discrete_features = (
                remap_discrete_features(self.feature_idcs, self.discrete_features)
                if self.discrete_features is not None
                else None
            )

        # Best-first tree fitting
        if self.splitter == BEST:
            self.leaves.append(self.node)
            sufficient_impurity_decrease = True
            while sufficient_impurity_decrease:
                if self.max_leaf_nodes is not None:
                    if len(self.leaves) == self.max_leaf_nodes:
                        break
                    elif len(self.leaves) > self.max_leaf_nodes:
                        raise Exception(
                            "Somehow created too many leaves. Should never be here."
                            + str(self.max_leaf_nodes)
                        )

                sufficient_impurity_decrease = True
                best_leaf = None
                best_leaf_idx = None
                best_leaf_reduction = float("inf")

                # Iterate over leaves and decide which to split
                # TODO: Perhaps we should be randomly choosing which leaf to split with finite budget, so that each leaf
                #  can be assessed on equal footing. Or engineer budget such that a full tree can be made?

                for leaf_idx, leaf in enumerate(self.leaves):
                    # First check splittable conditions that don't require calculating best split
                    if not self.check_splittable_constraints(leaf):
                        leaf.is_splittable = False
                        continue

                    # num_queries for the leaf should be updated only if we're not caching
                    # Need to get this before call to .calculate_best_split() below
                    split_already_computed = leaf.best_reduction_computed
                    if self.remaining_budget is None:
                        # Runs solve_mab if not previously computed, which incurs cost!
                        reduction = leaf.calculate_best_split(self.remaining_budget)
                    else:
                        if self.remaining_budget > 0:
                            reduction = leaf.calculate_best_split(self.remaining_budget)
                        else:
                            break

                    # don't add queries if best split is already computed
                    # add number of queries we made if the best split is NOT already computed
                    if not split_already_computed:
                        self.num_queries += leaf.num_queries
                        if self.remaining_budget is not None:
                            self.remaining_budget -= leaf.num_queries

                    if leaf.is_splittable is None:
                        # Uses cached value of calculate_best_split so no additional cost
                        leaf.is_splittable = self.check_splittable(leaf)

                    if (
                        reduction is not None
                        and reduction < best_leaf_reduction
                        and leaf.is_splittable
                    ):
                        best_leaf = leaf
                        best_leaf_idx = leaf_idx
                        best_leaf_reduction = reduction

                if (
                    best_leaf_reduction is not None
                    and best_leaf_reduction < self.min_impurity_decrease
                ):
                    best_leaf.split()
                    self.num_splits += 1
                    split_leaf = self.leaves.pop(best_leaf_idx)

                    # this node is no longer a leaf
                    split_leaf.prediction_probs = None
                    split_leaf.predicted_label = None

                    # We need to check that the split node actually has children, because in the case the MAB identifies
                    # a bad arm as negative return when, in fact, it has 0 return and puts all the children on one side,
                    # the split node won't have any children
                    if split_leaf.left is not None and split_leaf.right is not None:
                        self.leaves.append(split_leaf.left)
                        self.leaves.append(split_leaf.right)
                else:
                    sufficient_impurity_decrease = False

                self.depth = self.get_depth()

        # Depth-first tree fitting
        elif self.splitter == DEPTH:
            raise Exception(
                "Budget tracking in recursive splitting is not yet supported. Are you sure you know what you're doing?"
            )
            self.recursive_split(self.node)
        else:
            raise Exception("Invalid splitter choice")

        if self.verbose:
            print("Fitting finished")

    def recursive_split(self, node: Node) -> None:
        """
        Recursively split nodes till the termination condition is satisfied

        :param node: A root node to be split recursively
        """
        node.is_splittable = self.check_splittable(node)
        self.num_queries += node.num_queries
        if not node.is_splittable:
            self.leaves.append(node)
        else:
            self.num_splits += 1
            node.calculate_best_split()
            node.split()
            self.recursive_split(node.left)
            self.recursive_split(node.right)

    def predict(self, datapoint: np.ndarray) -> Union[Tuple[int, np.ndarray], float]:
        """
        Classifier: calculate the predicted probabilities that the given datapoint belongs to each classifier
        Regressor: calculate the mean of all labels(targets)

        :param datapoint: datapoint to fit
        :return: the probabilities of the datapoint being each class label or the mean value of labels
        """
        assert (
            len(datapoint.shape) == 1
        ), "Can only call .predict() directly on a single datapoint"
        node = self.node
        while node.left:
            feature_value = datapoint[node.split_feature]
            node = node.left if feature_value <= node.split_value else node.right
        assert node.right is None, "Tree is malformed"

        if self.is_classification:
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
        else:
            if node.predicted_value is not None:
                return node.predicted_value
            node.predicted_value = np.mean(node.labels)
            return float(node.predicted_value)

    def tree_print(self) -> None:
        """
        Print the tree depth-first in a format matching sklearn
        """
        self.node.n_print()
        print("\n")  # For consistency with sklearn

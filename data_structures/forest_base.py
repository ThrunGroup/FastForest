import numpy as np
from typing import Tuple, DefaultDict, Union
from abc import ABC
from collections import defaultdict

from utils.constants import (
    BUFFER,
    MAB,
    LINEAR,
    IDENTITY,
    GINI,
    BEST,
    MAX_SEED,
    DEFAULT_NUM_BINS,
    DEFAULT_REGRESSOR_LOSS,
    DEFAULT_MIN_IMPURITY_DECREASE,
)
from utils.utils import data_to_discrete, set_seed
from utils.boosting import get_next_targets
from data_structures.tree_classifier import TreeClassifier
from data_structures.tree_regressor import TreeRegressor


class ForestBase(ABC):
    """
    Class for vanilla random forest base model, which averages each tree's predictions
    """

    def __init__(
        self,
        data: np.ndarray = None,
        labels: np.ndarray = None,
        n_estimators: int = 100,
        max_depth: int = None,
        bootstrap: bool = True,
        feature_subsampling: str = None,
        tree_global_feature_subsampling: bool = False,
        min_samples_split: int = 2,
        min_impurity_decrease: float = DEFAULT_MIN_IMPURITY_DECREASE,
        max_leaf_nodes: int = None,
        bin_type: str = LINEAR,
        num_bins: int = DEFAULT_NUM_BINS,
        budget: int = None,
        criterion: str = GINI,
        splitter: str = BEST,
        solver: str = MAB,
        is_classification: bool = True,
        random_state: int = 0,
        with_replacement: bool = False,
        verbose: bool = False,
        boosting: bool = False,
        boosting_lr: float = None,
        make_discrete: bool = False,
    ) -> None:
        self.data = data
        self.org_targets = labels
        self.new_targets = labels

        # self.curr_data and self.curr_targets are the data, targets that are used to fit the current tree.
        # These attributes may be smaller than the original dataset size if self.bootstrap is true.
        self.curr_data = None
        self.curr_targets = None
        self.trees = []
        self.n_estimators = n_estimators
        self.is_classification = is_classification
        self.make_discrete = make_discrete
        self.discrete_features = None
        if (bin_type == LINEAR) or (bin_type == IDENTITY):
            self.make_discrete = False

        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.feature_subsampling = feature_subsampling
        self.tree_global_feature_subsampling = tree_global_feature_subsampling

        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_leaf_nodes = max_leaf_nodes
        self.bin_type = bin_type
        self.num_bins = num_bins

        self.remaining_budget = budget
        self.num_queries = 0

        self.criterion = criterion
        self.splitter = splitter
        self.solver = solver

        self.random_state = random_state
        set_seed(self.random_state)
        self.rng = np.random.default_rng(self.random_state)

        self.with_replacement = with_replacement
        self.verbose = verbose

        self.boosting = boosting
        if self.boosting and boosting_lr is None:
            raise Exception("Need to set boosting_lr when using boosting")
        if self.boosting and self.is_classification:
            raise NotImplementedError(
                "Boosting in classification is not supported yet."
            )
        self.boosting_lr = boosting_lr

        # Same parameters as sklearn.ensembleRandomForestClassifier. We won't need all of them.
        # See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

        self.min_samples_leaf = 1
        self.min_weight_fraction_leaf = 0.0
        self.max_features = None
        self.oob_score = False
        self.n_jobs = None
        self.warm_start = False
        self.class_weight = None
        self.ccp_alpha = 0.0
        self.max_samples = None

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
        Fit the random forest classifier by training trees, where each tree is trained with only a subset of the
        available features

        :return: None
        """
        self.check_both_or_neither(data, labels)
        if data is not None:
            self.data = data
            self.org_targets = labels
            self.new_targets = labels
        if self.make_discrete:
            self.discrete_features: DefaultDict = data_to_discrete(self.data, n=10)
        self.trees = []

        for i in range(self.n_estimators):
            if self.remaining_budget is not None and self.remaining_budget <= 0:
                break

            if self.verbose:
                print("Fitting tree", i)

            # If we're not boosting, fit to the original targets
            # If we're using boosting, set to the new targets that were computed at the end of the previous tree fit
            self.curr_targets = self.new_targets if self.boosting else self.org_targets

            if self.bootstrap:
                N = len(self.curr_targets)
                idcs = self.rng.choice(N, size=N, replace=True)
                self.curr_data = self.data[idcs]
                self.curr_targets = self.curr_targets[idcs]
            else:
                self.curr_data = self.data

            # NOTE: We cannot just let the tree's random states be forest.random_state + i, because then
            # two forests whose index is off by 1 will have very correlated results (e.g. when running multiple exps),
            # e.g., the first tree of the second forest will have the same random seed as the second tree of the first
            # forest. For this reason, we need to generate a new sequence of random numbers to seed the trees.
            tree_random_state = np.random.randint(MAX_SEED)

            if self.is_classification:
                tree = TreeClassifier(
                    data=self.curr_data,
                    labels=self.curr_targets,
                    max_depth=self.max_depth,
                    classes=self.classes,
                    budget=self.remaining_budget,
                    min_samples_split=self.min_samples_split,
                    min_impurity_decrease=self.min_impurity_decrease,
                    max_leaf_nodes=self.max_leaf_nodes,
                    discrete_features=self.discrete_features,
                    bin_type=self.bin_type,
                    num_bins=self.num_bins,
                    solver=self.solver,
                    feature_subsampling=self.feature_subsampling,
                    random_state=tree_random_state,
                    with_replacement=self.with_replacement,
                    verbose=self.verbose,
                    make_discrete=False,
                )
            else:
                tree = TreeRegressor(
                    data=self.curr_data,
                    labels=self.curr_targets,
                    max_depth=self.max_depth,
                    budget=self.remaining_budget,
                    min_samples_split=self.min_samples_split,
                    min_impurity_decrease=self.min_impurity_decrease,
                    max_leaf_nodes=self.max_leaf_nodes,
                    discrete_features=self.discrete_features,
                    bin_type=self.bin_type,
                    num_bins=self.num_bins,
                    solver=self.solver,
                    feature_subsampling=self.feature_subsampling,
                    random_state=tree_random_state,
                    with_replacement=self.with_replacement,
                    verbose=self.verbose,
                    make_discrete=False,
                )
            tree.fit()
            self.trees.append(tree)

            if self.boosting:
                # TODO: currently uses O(n) computation
                self.new_targets = get_next_targets(
                    loss_type=DEFAULT_REGRESSOR_LOSS,
                    is_classification=self.is_classification,
                    targets=self.new_targets,
                    predictions=self.predict_batch(self.data),
                )

            # Bookkeeping
            self.num_queries += tree.num_queries
            if self.remaining_budget is not None:
                self.remaining_budget -= tree.num_queries
                assert self.remaining_budget > -BUFFER, "Error: went over budget"

    def predict(self, datapoint: np.ndarray) -> Union[Tuple[int, np.ndarray], float]:
        """
        Generate a prediction for the given datapoint by averaging over all trees' predictions

        :param datapoint: datapoint to predict
        :returns: (Classifier) the averaged probabilities of the datapoint being each class label in each tree
                  and the class label with a greatest probability
                  (Regressor) the averaged mean value of labels in each tree
        """
        T = len(self.trees)
        if self.is_classification:
            agg_preds = np.empty((T, self.n_classes))
            for tree_idx, tree in enumerate(self.trees):
                # Average over predicted probabilities, not just hard labels
                agg_preds[tree_idx] = tree.predict(datapoint)[1]
            avg_preds = agg_preds.mean(axis=0)
            label_pred = list(self.classes.keys())[avg_preds.argmax()]
            return label_pred, avg_preds
        else:
            if self.boosting:
                for tree_idx, tree in enumerate(self.trees):
                    if tree_idx == 0:
                        agg_pred = tree.predict(datapoint)
                    else:
                        agg_pred += self.boosting_lr * tree.predict(datapoint)
                return agg_pred
            else:
                agg_pred = np.empty(T)
                for tree_idx, tree in enumerate(self.trees):
                    agg_pred[tree_idx] = tree.predict(datapoint)
                return float(agg_pred.mean())
